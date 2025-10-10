@echo off
setlocal EnableExtensions

REM Run the build via Python to avoid CMD parsing issues
set "SCRIPT_PATH=%~dp0"
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

REM Prefer venv Python if it exists (..\ .venv\Scripts\python.exe under mesa\)
set "VENV_PY=%SCRIPT_PATH%\..\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%SCRIPT_PATH%\build_all.py"
) else (
  python "%SCRIPT_PATH%\build_all.py"
)

endlocal
