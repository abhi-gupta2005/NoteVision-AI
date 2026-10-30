#!/usr/bin/env bash
# NoteVision AI - environment setup for macOS / Linux.
#   bash scripts/setup.sh            (uses python3.11 if available, else python3)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.11 python3.10 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "No python3 found"; exit 1; }
echo "Using interpreter: $PY ($($PY --version))"

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
if [ ! -d .vscode ] && [ -d vscode-config ]; then mkdir .vscode && cp vscode-config/*.json .vscode/; fi
python -c "import sys, librosa, streamlit; from src.basic_pitch_onnx import is_available; ok, d = is_available(); print('OK librosa', librosa.__version__, '|', d); sys.exit(0 if ok else 1)"
echo
echo "Done. Start with:  source .venv/bin/activate && streamlit run app.py"
