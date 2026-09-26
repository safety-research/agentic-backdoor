# Local demo

The optional interface shows a streamed conversation, proposed commands, their
terminal output, and the same target-inclusion metric used by the evaluation
pipeline. It is separate from the paper's evaluation and training workflows.

## Scripted illustration (CPU only)

From the repository root, use Python 3.10+ and Node.js 22.12+ with npm:

```bash
python -m venv /tmp/agentic-demo-env
source /tmp/agentic-demo-env/bin/activate
python -m pip install -r requirements/demo.txt
npm --prefix demo/frontend ci
npm --prefix demo/frontend run build
bash demo/run.sh
```

Open `http://127.0.0.1:8899`. Select an example and press Start. The default
`DEMO_MOCK=1` uses deterministic scripted responses and simulated command output;
it does not load a model, execute commands, access API keys, or contact 1Password.
The banner and command results label this mode explicitly. The scripted trigger
example illustrates a target match; it is not evidence of model behavior or ASR.

The target command is read from `src/common/recipe.py`; its default URL is the
reserved placeholder `https://example.invalid/payload.sh`. No payload is bundled.
The paired filesystem examples are synthetic and are not evaluation prompts.

## Live checkpoint

Use the configured `eval` environment with vLLM, install
`requirements/demo.txt` and `udocker`, and build the frontend as above. Provide
a local Hugging Face checkpoint and a container image with bash and the tools
needed for your tasks:

```bash
DEMO_MOCK=0 \
DEMO_MODEL_PATH=/path/to/hf-checkpoint \
DEMO_CONTAINER_IMAGE=your-prepared-image:tag \
bash demo/run.sh
```

For multiple checkpoints, copy `models.example.json` to `models.local.json`,
edit its paths, and set `DEMO_MODELS_FILE=demo/models.local.json` instead of
`DEMO_MODEL_PATH`. Local configuration is ignored by Git. Point at an exported
Hugging Face checkpoint, including its `config.json` and model weights; automatic
discovery of experiment directories is intentionally absent.

Live mode executes model-generated commands. Run it in a disposable research
VM without credentials or sensitive mounts, with network access restricted by
the host. udocker is a convenience runtime, not a security boundary. The web
server binds to loopback and supports one active session. For a remote machine,
use an SSH tunnel, for example `ssh -L 8899:127.0.0.1:8899 your-host`. GPU resource
allocation follows your scheduler; the demo does not submit or cancel jobs.

The live demo uses illustrative, multi-turn generation settings from
`backend/config.py`. For paper metrics, use the evaluation pipeline and its
held-out prompts and sampling profiles instead. Local inference requires no
Anthropic API key. Changing the scoring target does not change a trained model's
behavior; original checkpoint outputs may therefore not match the public target.

## Development and checks

Rebuild the frontend after changing it. For hot reload, run the backend on its
default port and `npm --prefix demo/frontend run dev` in a second terminal;
open `http://127.0.0.1:5173`. `DEMO_PORT` changes the built demo's backend port.

```bash
python -m unittest discover -s demo/tests -v
npm --prefix demo/frontend run build
```
