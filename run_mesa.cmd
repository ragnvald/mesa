@echo off
REM Double-click launcher for MESA. Runs the repo's .venv (Python 3.14) on mesa.py.
REM Paths are relative to this file (%~dp0), so it works from any clone / location
REM and is independent of the .py file association (Wing IDE etc.).
setlocal
set "HERE=%~dp0"
set "PYW=%HERE%.venv\Scripts\pythonw.exe"

if not exist "%PYW%" (
  echo [run_mesa] Python venv not found:
  echo     %PYW%
  echo.
  echo Create it with Python 3.14:  devtools\setup_venvs.bat
  echo.
  pause
  exit /b 1
)

REM Launch windowless ^(GUI app^); the .cmd window closes immediately.
start "MESA" "%PYW%" "%HERE%mesa.py"
endlocal
