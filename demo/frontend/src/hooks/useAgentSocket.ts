import { useCallback, useEffect, useRef, useState } from "react";
import type { Terminal, IMarker } from "@xterm/xterm";
import type { AgentState, ClientMessage, ConversationEntry, ServerMessage } from "../types";

const WS_URL = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/agent`;

export function useAgentSocket(agentTerminal?: Terminal | null) {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<AgentState>("idle");
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);

  // Ref for agentTerminal — avoids re-creating the WS when the terminal mounts
  const terminalRef = useRef<Terminal | null>(null);
  terminalRef.current = agentTerminal ?? null;

  // Streaming token buffer — flushed via rAF
  const streamBuf = useRef("");
  const flushScheduled = useRef(false);
  const streamingActive = useRef(false);

  // Terminal markers for click-to-scroll
  const nextCommandId = useRef(0);
  const markerMap = useRef<Map<number, IMarker>>(new Map());

  const flushTokens = useCallback(() => {
    flushScheduled.current = false;
    if (!streamingActive.current) return;
    const text = streamBuf.current;
    setConversation((c) => {
      const last = c[c.length - 1];
      if (last?.kind === "assistant" && last.streaming) {
        return [...c.slice(0, -1), { ...last, content: text }];
      }
      return [...c, { kind: "assistant", content: text, think: null, streaming: true }];
    });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (!flushScheduled.current) {
      flushScheduled.current = true;
      requestAnimationFrame(flushTokens);
    }
  }, [flushTokens]);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    try {
      const sock = new WebSocket(WS_URL);
      ws.current = sock;

      sock.onopen = () => setConnected(true);
      sock.onerror = () => setConnected(false);
      sock.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };

      sock.onmessage = (ev) => {
        const msg: ServerMessage = JSON.parse(ev.data);

        switch (msg.type) {
          case "status":
            setState(msg.state);
            // Clear terminal and write initial prompt for new conversation
            if (msg.state === "preparing_container" && terminalRef.current) {
              terminalRef.current.clear();
              terminalRef.current.reset();
              terminalRef.current.write("\x1b[1;32mAGENT\x1b[0m$ ");
            }
            break;

          case "system_prompt":
            setConversation((c) => {
              const filtered = c.filter(
                (e) => !(e.kind === "info" && e.content === "Starting agent...")
              );
              return [{ kind: "system", content: msg.content }, ...filtered];
            });
            break;

          case "user_message":
            break;

          case "token":
            streamBuf.current += msg.text;
            streamingActive.current = true;
            scheduleFlush();
            break;

          case "assistant_complete":
            streamBuf.current = "";
            streamingActive.current = false;
            flushScheduled.current = false;
            setConversation((c) => {
              const last = c[c.length - 1];
              if (last?.kind === "assistant" && last.streaming) {
                if (!msg.content.trim()) return c.slice(0, -1);
                return [
                  ...c.slice(0, -1),
                  { kind: "assistant", content: msg.content, think: msg.think, streaming: false },
                ];
              }
              if (!msg.content.trim()) return c;
              return [
                ...c,
                { kind: "assistant", content: msg.content, think: msg.think, streaming: false },
              ];
            });
            break;

          case "command_start": {
            const cmdId = nextCommandId.current++;
            if (terminalRef.current) {
              // Register marker at current cursor line (the AGENT$ prompt line)
              const marker = terminalRef.current.registerMarker(0);
              if (marker) {
                markerMap.current.set(cmdId, marker);
              }
              // Write command text after the existing AGENT$ prompt, then newline
              terminalRef.current.write(msg.command + "\r\n");
            }
            setConversation((c) => [
              ...c,
              {
                kind: "command",
                command: msg.command,
                target_match: msg.target_match,
                done: false,
                commandId: cmdId,
              },
            ]);
            break;
          }

          case "command_output":
            // Write command output to the terminal, then the next prompt
            if (terminalRef.current) {
              const parts: string[] = [];
              if (msg.stdout) parts.push(msg.stdout);
              if (msg.stderr) parts.push(msg.stderr);
              const output = parts.join("\n").trim();
              if (output) {
                // Normalize line endings for xterm (\n → \r\n)
                terminalRef.current.write(
                  output.replace(/\r\n/g, "\n").replace(/\n/g, "\r\n") + "\r\n"
                );
              }
              if (msg.timed_out) {
                terminalRef.current.write("\x1b[33m[TIMEOUT]\x1b[0m\r\n");
              }
              terminalRef.current.write("\x1b[1;32mAGENT\x1b[0m$ ");
            }
            setConversation((c) => {
              const idx = c.findLastIndex(
                (e: ConversationEntry) =>
                  e.kind === "command" && e.command === msg.command && !e.done
              );
              if (idx < 0) return c;
              const updated = {
                ...c[idx],
                stdout: msg.stdout,
                stderr: msg.stderr,
                returncode: msg.returncode,
                timed_out: msg.timed_out,
                simulated: msg.simulated,
                done: true,
              } as ConversationEntry;
              return [...c.slice(0, idx), updated, ...c.slice(idx + 1)];
            });
            break;

          case "turn_complete":
            break;

          case "conversation_end":
            streamingActive.current = false;
            setConversation((c) => {
              const last = c[c.length - 1];
              const cleaned =
                last?.kind === "assistant" && last.streaming ? c.slice(0, -1) : c;
              return [...cleaned, { kind: "info", content: `Conversation ended: ${msg.reason}` }];
            });
            setState("idle");
            break;

          case "error":
            setConversation((c) => [
              ...c,
              { kind: "info", content: `Error: ${msg.message}` },
            ]);
            setState("idle");
            break;
        }
      };
    } catch {
      setTimeout(connect, 3000);
    }
  }, [scheduleFlush]);

  useEffect(() => {
    connect();
    return () => ws.current?.close();
  }, [connect]);

  const send = useCallback((msg: ClientMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(msg));
    }
  }, []);

  const startConversation = useCallback(
    (task: string, env: string, model: string, sysPrompt?: string) => {
      // Clean up old markers
      for (const marker of markerMap.current.values()) {
        marker.dispose();
      }
      markerMap.current.clear();
      nextCommandId.current = 0;

      streamBuf.current = "";
      streamingActive.current = false;
      flushScheduled.current = false;

      // Clear terminal and write initial prompt
      if (terminalRef.current) {
        terminalRef.current.clear();
        terminalRef.current.reset();
        terminalRef.current.write("\x1b[1;32mAGENT\x1b[0m$ ");
      }

      setConversation([
        { kind: "user", content: task, source: "user" },
        { kind: "info", content: "Starting agent..." },
      ]);
      setState("loading_model");
      send({
        type: "start",
        task,
        env,
        model,
        ...(sysPrompt ? { sys_prompt: sysPrompt } : {}),
      } as ClientMessage);
    },
    [send]
  );

  const stop = useCallback(() => send({ type: "stop" }), [send]);

  const injectMessage = useCallback(
    (content: string) => {
      setConversation((c) => [...c, { kind: "user", content, source: "human" }]);
      send({ type: "inject_message", content });
    },
    [send]
  );

  const scrollToCommand = useCallback(
    (entry: ConversationEntry) => {
      const term = terminalRef.current;
      if (entry.kind !== "command" || entry.commandId == null || !term) return;
      const marker = markerMap.current.get(entry.commandId);
      if (marker && !marker.isDisposed && marker.line >= 0) {
        // Scroll with 2 lines of context above so the command isn't jammed at the top
        term.scrollToLine(Math.max(0, marker.line - 2));
        // Briefly select just the command line to highlight it
        const viewportLine = marker.line - term.buffer.active.viewportY;
        if (viewportLine >= 0) {
          term.select(0, viewportLine, term.cols);
          setTimeout(() => term.clearSelection(), 1500);
        }
      }
    },
    []
  );

  return {
    connected,
    state,
    conversation,
    startConversation,
    stop,
    injectMessage,
    scrollToCommand,
  };
}
