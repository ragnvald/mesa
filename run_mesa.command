#!/bin/bash
# Double-click launcher for MESA on macOS. Runs the repo's .venv (Python 3.14)
# on mesa.py. Paths are relative to this file, so it works from any clone /
# location. macOS counterpart of run_mesa.cmd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python3"

if [ ! -x "$PY" ]; then
  echo "[run_mesa] Python venv not found:"
  echo "    $PY"
  echo
  echo "Create it with Python 3.14 (see devtools/setup_venvs.bat on Windows,"
  echo "or python3.14 -m venv .venv && pip install -r requirements_py314.txt)."
  echo
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

# Launch detached so the Terminal window can close without killing the GUI.
cd "$HERE"
nohup "$PY" "$HERE/mesa.py" >/dev/null 2>&1 &
