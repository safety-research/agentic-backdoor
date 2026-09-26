"""Local, single-session demo API and streamed conversation interface."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent_loop import AgentLoop
from .config import (
    CONTAINER_IMAGE, DEFAULT_ENV, DEFAULT_MODEL, ENVS, EXAMPLE_TASKS, MOCK_MODE, MODELS,
)
from .container_manager import ContainerManager

log = logging.getLogger(__name__)
if MOCK_MODE:
    from .mock_model import MockModelManager
    model_manager = MockModelManager()
else:
    if not MODELS or not CONTAINER_IMAGE:
        raise RuntimeError("Live demo requires DEMO_MODEL_PATH (or DEMO_MODELS_FILE) and DEMO_CONTAINER_IMAGE")
    from .model_manager import ModelManager
    model_manager = ModelManager()

container_manager = ContainerManager()
session_lock = asyncio.Lock()
port = os.environ.get("DEMO_PORT", "8899")
ALLOWED_ORIGINS = {
    f"http://{host}:{allowed_port}"
    for host in ("127.0.0.1", "localhost")
    for allowed_port in (port, "5173")
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await model_manager.load_model(DEFAULT_MODEL)
        if not MOCK_MODE:
            await container_manager.create(DEFAULT_ENV)
        yield
    finally:
        if not MOCK_MODE:
            await container_manager.destroy()


app = FastAPI(title="Agentic Backdoor Demo", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.add_middleware(CORSMiddleware, allow_origins=sorted(ALLOWED_ORIGINS),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def check_origin(request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"error": "Origin is not allowed"}, status_code=403)
    return await call_next(request)


@app.get("/api/status")
async def status():
    return {"mode": "scripted" if MOCK_MODE else "live", "executes_commands": not MOCK_MODE}


@app.get("/api/models")
async def list_models():
    return {"models": {key: value["label"] for key, value in MODELS.items()},
            "current": model_manager.current_model_id}


@app.get("/api/envs")
async def list_envs():
    return {"envs": {key: value["label"] for key, value in ENVS.items()}}


@app.get("/api/examples")
async def list_examples():
    return {"examples": EXAMPLE_TASKS}


@app.post("/api/reset-container/{env_type}")
async def reset_container(env_type: str):
    if env_type not in ENVS:
        return JSONResponse({"error": "Unknown environment"}, status_code=400)
    if session_lock.locked():
        return JSONResponse({"error": "A session is running"}, status_code=409)
    async with session_lock:
        if not MOCK_MODE:
            await container_manager.reset(env_type, force=True)
    return {"status": "ok", "env": env_type, "simulated": MOCK_MODE}


@app.websocket("/ws/agent")
async def agent_websocket(ws: WebSocket):
    origin = ws.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await ws.close(code=1008)
        return
    await ws.accept()
    agent = None
    run_task = None
    owns_session = False

    async def run_conversation(message: dict, loop: AgentLoop):
        nonlocal owns_session
        try:
            model_id = message.get("model") or DEFAULT_MODEL
            if model_id != model_manager.current_model_id:
                await ws.send_json({"type": "status", "state": "loading_model"})
                await model_manager.load_model(model_id)
            await ws.send_json({"type": "status", "state": "preparing_container"})
            if not MOCK_MODE:
                await container_manager.reset(message.get("env", DEFAULT_ENV))
                await container_manager.spawn_agent_pty()
            async for event in loop.run(message["task"], message.get("env", DEFAULT_ENV),
                                        sys_prompt=message.get("sys_prompt")):
                await ws.send_json(event.to_dict())
        except asyncio.CancelledError:
            await model_manager.abort_current()
            raise
        except Exception as error:
            log.exception("Demo session failed")
            await ws.send_json({"type": "error", "message": str(error)})
            await ws.send_json({"type": "status", "state": "idle"})
        finally:
            session_lock.release()
            owns_session = False

    try:
        while True:
            try:
                message = await ws.receive_json()
            except ValueError:
                await ws.send_json({"type": "error", "message": "Expected a JSON object"})
                continue
            if not isinstance(message, dict):
                await ws.send_json({"type": "error", "message": "Expected a JSON object"})
                continue
            kind = message.get("type")
            if kind == "start":
                if session_lock.locked():
                    await ws.send_json({"type": "error", "message": "A session is already running"})
                    continue
                task = message.get("task")
                env_type = message.get("env", DEFAULT_ENV)
                model_id = message.get("model") or DEFAULT_MODEL
                sys_prompt = message.get("sys_prompt")
                if (
                    not isinstance(task, str) or not task.strip()
                    or not isinstance(env_type, str) or env_type not in ENVS
                    or not isinstance(model_id, str) or model_id not in MODELS
                    or (sys_prompt is not None and not isinstance(sys_prompt, str))
                ):
                    await ws.send_json({"type": "error", "message": "Provide a task and a listed model/environment"})
                    continue
                await session_lock.acquire()
                owns_session = True
                agent = AgentLoop(model_manager, container_manager, mock_mode=MOCK_MODE)
                run_task = asyncio.create_task(run_conversation(message, agent))
            elif kind == "stop" and agent is not None:
                agent.cancel()
                await model_manager.abort_current()
            elif kind == "inject_message" and agent is not None:
                content = message.get("content")
                if isinstance(content, str) and content:
                    agent.inject_message(content)
    except WebSocketDisconnect:
        pass
    finally:
        if agent is not None:
            agent.cancel()
        if run_task is not None:
            if not run_task.done():
                run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        if owns_session:
            # Cancellation can happen before the background task starts.
            session_lock.release()


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
