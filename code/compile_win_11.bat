@echo off
setlocal EnableExtensions

REM Kjor bygg via Python for å unngå CMD-parsing-problemer
set "SCRIPT_PATH=%~dp0"
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

REM Foretrekk venv-Python hvis den finnes
set "VENV_PY=%SCRIPT_PATH%\..\ .venv\Scripts\python.exe"
set "VENV_PY=%SCRIPT_PATH%\..\ .venv\Scripts\python.exe"

if exist "%SCRIPT_PATH%\..\ .venv\Scripts\python.exe" (
  "%SCRIPT_PATH%\..\ .venv\Scripts\python.exe" "%SCRIPT_PATH%\build_all.py"
) else (
  python "%SCRIPT_PATH%\build_all.py"
)

endlocal
