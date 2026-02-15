@echo off
setlocal EnableExtensions

REM Force a full build (main + all helpers) when running this script with no parameters.
REM This avoids surprises if the user environment has lingering MESA_* overrides.
set "MESA_BUILD_MAIN=1"
set "MESA_BUILD_HELPERS=1"
set "MESA_BUILD_CLEAN=1"
set "MESA_HELPERS="
set "MESA_HELPERS_SKIP="

REM Optional: pass "fast" to skip PyInstaller clean (faster rebuilds, less reliable if deps changed)
if /I "%~1"=="fast" (
  set "MESA_BUILD_CLEAN=0"
)

echo [MESA] Full build enforced: MESA_BUILD_MAIN=1, MESA_BUILD_HELPERS=1, MESA_BUILD_CLEAN=%MESA_BUILD_CLEAN%

REM Capture start time (FileTime ticks) to compute total duration later
for /f %%I in ('powershell -NoLogo -Command "(Get-Date).ToFileTimeUtc()"') do set "START_TICKS=%%I"

REM Run the build via Python to avoid CMD parsing issues
set "SCRIPT_PATH=%~dp0"
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

REM Prefer venv Python if it exists (..\.venv\Scripts\python.exe under mesa\)
set "VENV_PY=%SCRIPT_PATH%\..\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%SCRIPT_PATH%\build_all.py"
) else (
  python "%SCRIPT_PATH%\build_all.py"
)

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
