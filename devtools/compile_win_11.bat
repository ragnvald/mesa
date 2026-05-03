@echo off
setlocal EnableExtensions

REM Force a full build (main + all helpers) when running this script with no parameters.
REM This avoids surprises if the user environment has lingering MESA_* overrides.
set "MESA_BUILD_MAIN=1"
set "MESA_BUILD_HELPERS=1"
set "MESA_BUILD_CLEAN=1"
set "MESA_HELPERS="
set "MESA_HELPERS_SKIP="

REM Default to 4-way parallel (5 helpers + main = 6 PyInstaller invocations).
REM Override with the env var or with the "serial" / "parallel N" arg below.
if not defined MESA_BUILD_PARALLEL set "MESA_BUILD_PARALLEL=4"

REM Optional: pass "fast" to skip PyInstaller clean (faster rebuilds, less reliable if deps changed)
if /I "%~1"=="fast" (
  set "MESA_BUILD_CLEAN=0"
)

REM Optional: pass "serial" to force sequential builds (safer on memory-tight hosts).
if /I "%~1"=="serial" (
  set "MESA_BUILD_PARALLEL=1"
)

REM Optional: pass "parallel N" to override the worker count.
if /I "%~1"=="parallel" if not "%~2"=="" (
  set "MESA_BUILD_PARALLEL=%~2"
)

echo [MESA] Full build enforced: MESA_BUILD_MAIN=1, MESA_BUILD_HELPERS=1, MESA_BUILD_CLEAN=%MESA_BUILD_CLEAN%, MESA_BUILD_PARALLEL=%MESA_BUILD_PARALLEL%

REM Capture start time (FileTime ticks) to compute total duration later
for /f %%I in ('powershell -NoLogo -Command "(Get-Date).ToFileTimeUtc()"') do set "START_TICKS=%%I"

REM Run the build via Python to avoid CMD parsing issues
set "SCRIPT_PATH=%~dp0"
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

REM Hide pygame startup banner if pygame is present in fallback environments.
set "PYGAME_HIDE_SUPPORT_PROMPT=1"

REM Python selection for compile builds (in priority order):
REM  1) MESA_COMPILE_PYTHON (explicit override)
REM  2) ..\.venv_compile\Scripts\python.exe  (recommended)
REM  3) ..\.venv\Scripts\python.exe          (development fallback)
REM  4) python from PATH
set "VENV_COMPILE_PY=%SCRIPT_PATH%\..\.venv_compile\Scripts\python.exe"
set "VENV_DEV_PY=%SCRIPT_PATH%\..\.venv\Scripts\python.exe"
set "BUILD_PYTHON="

if defined MESA_COMPILE_PYTHON (
  set "BUILD_PYTHON=%MESA_COMPILE_PYTHON%"
) else if exist "%VENV_COMPILE_PY%" (
  set "BUILD_PYTHON=%VENV_COMPILE_PY%"
) else if exist "%VENV_DEV_PY%" (
  set "BUILD_PYTHON=%VENV_DEV_PY%"
) else (
  set "BUILD_PYTHON=python"
)

echo [MESA] Compile Python: %BUILD_PYTHON%
"%BUILD_PYTHON%" -c "import sys; print('[MESA] Python version:', sys.version.split()[0])"
if errorlevel 1 (
  echo [MESA] Failed to start selected Python: %BUILD_PYTHON%
  endlocal & exit /b 1
)

"%BUILD_PYTHON%" "%SCRIPT_PATH%\build_all.py"

set "BUILD_EXITCODE=%ERRORLEVEL%"

REM Compute elapsed time (format HH:MM:SS, hours may exceed 24) alongside start/end timestamps
for /f "usebackq tokens=1-3 delims=|" %%I in (`
  powershell -NoLogo -Command ^
    "$startUtc=[datetime]::FromFileTimeUtc([int64]$env:START_TICKS);" ^
    "$endUtc=(Get-Date).ToUniversalTime();" ^
    "$span=New-TimeSpan -Start $startUtc -End $endUtc;" ^
    "$startLocal=$startUtc.ToLocalTime();" ^
    "$endLocal=$endUtc.ToLocalTime();" ^
    "'{0:yyyy-MM-dd HH:mm:ss}|{1:yyyy-MM-dd HH:mm:ss}|{2:00}:{3:00}:{4:00}' -f $startLocal, $endLocal, [int][math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds"
`) do (
  set "START_TIME=%%I"
  set "END_TIME=%%J"
  set "ELAPSED=%%K"
)

echo Build started at %START_TIME%
echo Build finished at %END_TIME%

if "%BUILD_EXITCODE%"=="0" (
  echo Build completed successfully in %ELAPSED%
) else (
  echo Build failed ^(exit code %BUILD_EXITCODE%^) after %ELAPSED%
)

endlocal & exit /b %BUILD_EXITCODE%
