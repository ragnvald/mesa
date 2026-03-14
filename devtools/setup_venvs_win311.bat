@echo off
setlocal EnableExtensions

set "SCRIPT_PATH=%~dp0"
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"
set "REPO_ROOT=%SCRIPT_PATH%\.."

set "DEV_VENV=%REPO_ROOT%\.venv"
set "COMPILE_VENV=%REPO_ROOT%\.venv_compile"

set "DEV_REQ=%REPO_ROOT%\requirements_all_win311.txt"
set "COMPILE_REQ=%REPO_ROOT%\requirements_compile_win311.txt"

call :setup_one "%DEV_VENV%" "%DEV_REQ%" "development"
if errorlevel 1 (
  endlocal & exit /b 1
)

call :setup_one "%COMPILE_VENV%" "%COMPILE_REQ%" "compile"
if errorlevel 1 (
  endlocal & exit /b 1
)

echo.
echo [MESA] Environment setup complete.
echo [MESA] Development venv: %DEV_VENV%
echo [MESA] Compile venv:     %COMPILE_VENV%
echo.
echo Activate development venv:
echo   %DEV_VENV%\Scripts\Activate.ps1
echo.
echo Activate compile venv:
echo   %COMPILE_VENV%\Scripts\Activate.ps1
echo.

endlocal & exit /b 0

:setup_one
set "TARGET_VENV=%~1"
set "REQ_FILE=%~2"
set "LABEL=%~3"

if not exist "%REQ_FILE%" (
  echo [ERROR] Requirements file not found for %LABEL% venv: %REQ_FILE%
  exit /b 1
)

if exist "%TARGET_VENV%\Scripts\python.exe" (
  echo [MESA] Reusing %LABEL% venv: %TARGET_VENV%
) else (
  echo [MESA] Creating %LABEL% venv: %TARGET_VENV%
  py -3.11 -m venv "%TARGET_VENV%"
  if errorlevel 1 (
    echo [ERROR] Failed to create %LABEL% venv: %TARGET_VENV%
    exit /b 1
  )
)

set "TARGET_PY=%TARGET_VENV%\Scripts\python.exe"

echo [MESA] Upgrading pip in %LABEL% venv...
"%TARGET_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip in %LABEL% venv.
  exit /b 1
)

echo [MESA] Installing %LABEL% requirements from %REQ_FILE% ...
"%TARGET_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
  echo [ERROR] Failed installing %LABEL% requirements.
  exit /b 1
)

"%TARGET_PY%" -c "import sys; print('[MESA] Ready:', sys.executable)"
if errorlevel 1 (
  echo [ERROR] Verification failed for %LABEL% venv.
  exit /b 1
)

exit /b 0
