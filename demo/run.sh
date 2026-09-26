#!/usr/bin/env bash
# Run from an environment with requirements/demo.txt already installed.
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEMO_ROOT"
export DEMO_MOCK="${DEMO_MOCK:-1}"
export DEMO_PORT="${DEMO_PORT:-8899}"
export PYTHONPATH="$DEMO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -f demo/frontend/dist/index.html ]; then
    echo "Build the frontend first: npm --prefix demo/frontend ci && npm --prefix demo/frontend run build" >&2
    exit 1
fi
echo "Open http://127.0.0.1:${DEMO_PORT} (DEMO_MOCK=${DEMO_MOCK})"
exec python -m uvicorn demo.backend.server:app --host 127.0.0.1 --port "$DEMO_PORT"
