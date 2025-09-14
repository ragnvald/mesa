@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem (Optional) nice console, harmless if it fails
chcp 65001 >nul 2>&1

echo.
echo ==============================
echo   Compiling Python scripts...
echo ==============================
echo.

rem --- Paths ----------------------------------------------------------
set "SCRIPT_PATH=%~dp0"
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"
for %%i in ("%SCRIPT_PATH%") do set "PARENT_DIR=%%~dpi"

set "BUILD_FOLDER_ROOT=%PARENT_DIR%build"
set "DIST_FOLDER_ROOT=%PARENT_DIR%dist"
set "TOOLS_STAGE_DIR=%BUILD_FOLDER_ROOT%\tools_dist"
set "TOOLS_SRC_DIR=%SCRIPT_PATH%\tools"

echo SCRIPT_PATH          = %SCRIPT_PATH%
echo PROJECT ROOT         = %PARENT_DIR%
echo BUILD_FOLDER_ROOT    = %BUILD_FOLDER_ROOT%
echo DIST_FOLDER_ROOT     = %DIST_FOLDER_ROOT%
echo TOOLS_STAGE_DIR      = %TOOLS_STAGE_DIR%
echo TOOLS_SRC_DIR        = %TOOLS_SRC_DIR%
echo.

rem ===================================================================
rem   PRE-CLEAN: kill running EXEs that could lock dist, then wipe dist
rem ===================================================================

call :kill_if_running "mesa.exe"
call :kill_if_running "data_import.exe"
call :kill_if_running "assetgroup_edit.exe"
call :kill_if_running "geocodegroup_edit.exe"
call :kill_if_running "parameters_setup.exe"
call :kill_if_running "data_process.exe"
call :kill_if_running "atlas_edit.exe"
call :kill_if_running "atlas_create.exe"
call :kill_if_running "lines_admin.exe"
call :kill_if_running "lines_process.exe"
call :kill_if_running "data_report.exe"
call :kill_if_running "geocodes_create.exe"
call :kill_if_running "maps_overview.exe"

echo Cleaning dist folder...
call :clean_dir "%DIST_FOLDER_ROOT%"
if errorlevel 1 (
  echo ERROR: Could not clean %DIST_FOLDER_ROOT%. Close any running apps and retry.
  exit /b 1
)
mkdir "%DIST_FOLDER_ROOT%" >nul 2>&1

rem Ensure staging dirs exist
if not exist "%TOOLS_STAGE_DIR%"   mkdir "%TOOLS_STAGE_DIR%"
if not exist "%TOOLS_SRC_DIR%"     mkdir "%TOOLS_SRC_DIR%"

echo Copying resource folders to dist root...
xcopy "%SCRIPT_PATH%\system_resources" "%DIST_FOLDER_ROOT%\system_resources" /E /I /Y >nul 2>&1
xcopy "%SCRIPT_PATH%\input"            "%DIST_FOLDER_ROOT%\input"            /E /I /Y >nul 2>&1
xcopy "%SCRIPT_PATH%\output"           "%DIST_FOLDER_ROOT%\output"           /E /I /Y >nul 2>&1
xcopy "%SCRIPT_PATH%\qgis"             "%DIST_FOLDER_ROOT%\qgis"             /E /I /Y >nul 2>&1
xcopy "%SCRIPT_PATH%\docs"             "%DIST_FOLDER_ROOT%\docs"             /E /I /Y >nul 2>&1

echo.
echo Activating virtual environment...
call "%PARENT_DIR%.venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Could not activate venv at %PARENT_DIR%.venv
  exit /b 1
)
echo Venv OK.
echo.

rem --- Sanity check: mesa.py & icon ----------------------------------
if not exist "%SCRIPT_PATH%\mesa.py" (
  echo ERROR: mesa.py not found beside this script.
  exit /b 1
)
if not exist "%SCRIPT_PATH%\system_resources\mesa.ico" (
  echo WARNING: mesa.ico not found; proceeding without an icon.
  set "ICON_ARG="
) else (
  set "ICON_ARG=--icon=%SCRIPT_PATH%\system_resources\mesa.ico"
)

rem ===================================================================
rem   1) Build helper tools (GUI onefile EXEs) into TOOLS_STAGE_DIR
rem ===================================================================
echo Building helper tools...

call :build_one "data_import.py"
call :build_one "assetgroup_edit.py"
call :build_one "geocodegroup_edit.py"
call :build_one "parameters_setup.py"
call :build_one "data_process.py"
call :build_one "atlas_edit.py"
call :build_one "atlas_create.py"
call :build_one "lines_admin.py"
call :build_one "lines_process.py"
call :build_one "data_report.py"
call :build_one "geocodes_create.py"
call :build_one "maps_overview.py"

echo Staging helpers into %TOOLS_SRC_DIR% ...
del /Q "%TOOLS_SRC_DIR%\*.exe" >nul 2>&1
if exist "%TOOLS_STAGE_DIR%\*.exe" (
  copy /Y "%TOOLS_STAGE_DIR%\*.exe" "%TOOLS_SRC_DIR%\" >nul
  echo   [OK] helpers staged
) else (
  echo   [WW] no helper EXEs produced
)

rem ===================================================================
rem   2) Create mesa.spec that bundles system_resources + tools folder
rem ===================================================================

rem Clean any old spec
if exist "%SCRIPT_PATH%\mesa.spec" del /q "%SCRIPT_PATH%\mesa.spec" >nul 2>&1

echo Creating mesa.spec with pyi-makespec...
pyi-makespec ^
  --onefile --windowed ^
  %ICON_ARG% ^
  --collect-all ttkbootstrap ^
  --collect-all tkinterweb ^
  --hidden-import=ttkbootstrap ^
  --hidden-import=yaml ^
  --add-data "%SCRIPT_PATH%\system_resources;system_resources" ^
  --add-data "%TOOLS_SRC_DIR%;tools" ^
  "%SCRIPT_PATH%\mesa.py"

if errorlevel 1 (
  echo ERROR: pyi-makespec failed.
  exit /b 1
)

echo Patching recursion limit in mesa.spec...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = '%SCRIPT_PATH%\mesa.spec';" ^
  "$txt = Get-Content -LiteralPath $p -Raw;" ^
  "$line = 'import sys; sys.setrecursionlimit(sys.getrecursionlimit()*5)';" ^
  "if ($txt -notmatch [regex]::Escape($line)) { Set-Content -LiteralPath $p -Value ($line + \"`r`n\" + $txt) -NoNewline; }"

if errorlevel 1 (
  echo ERROR: Failed to patch mesa.spec.
  exit /b 1
)

echo.
echo Building mesa.exe from spec...
pyinstaller ^
  --noconfirm --clean ^
  --distpath "%DIST_FOLDER_ROOT%" ^
  --workpath "%BUILD_FOLDER_ROOT%" ^
  "%SCRIPT_PATH%\mesa.spec"

if errorlevel 1 (
  echo ERROR: pyinstaller failed for mesa.spec
  exit /b 1
)

rem --- Copy config.ini next to mesa.exe -------------------------------
echo.
echo Copying config.ini next to mesa.exe ...
if exist "%SCRIPT_PATH%\config.ini" (
  copy /Y "%SCRIPT_PATH%\config.ini" "%DIST_FOLDER_ROOT%\config.ini" >nul
) else (
  echo WARNING: config.ini not found at %SCRIPT_PATH%\config.ini
)

rem --- Verify output --------------------------------------------------
echo.
echo Verifying mesa.exe & config.ini in dist...
if exist "%DIST_FOLDER_ROOT%\mesa.exe" ( echo   [OK] mesa.exe ) else ( echo   [XX] mesa.exe missing & exit /b 1 )
if exist "%DIST_FOLDER_ROOT%\config.ini" ( echo   [OK] config.ini ) else ( echo   [WW] config.ini missing )

echo.
echo ==============================
echo   Build complete
echo   Dist: %DIST_FOLDER_ROOT%
echo ==============================
echo.

rem --- Cleanup build directory (optional) ----------------------------
if exist "%BUILD_FOLDER_ROOT%" (
  echo Removing %BUILD_FOLDER_ROOT%
  rmdir /s /q "%BUILD_FOLDER_ROOT%"
)

rem Keep repo tidy: remove staged helper EXEs from code\tools
del /Q "%TOOLS_SRC_DIR%\*.exe" >nul 2>&1

rem Remove spec in code folder
del /Q "%SCRIPT_PATH%\*.spec" >nul 2>&1

endlocal
exit /b 0


rem ===================================================================
rem Subroutine: kill_if_running  (arg1 = exe name, e.g. "mesa.exe")
rem ===================================================================
:kill_if_running
set "PROC=%~1"
if not defined PROC goto :eof
tasklist /FI "IMAGENAME eq %PROC%" | find /I "%PROC%" >nul
if not errorlevel 1 (
  echo   Closing %PROC% ...
  taskkill /IM %PROC% /F >nul 2>&1
  rem brief wait
  ping -n 2 127.0.0.1 >nul
)
goto :eof


rem ===================================================================
rem Subroutine: clean_dir  (arg1 = folder path)
rem Robust delete using PowerShell; retries to avoid locks.
rem ===================================================================
:clean_dir
set "DIR=%~1"
if not defined DIR exit /b 0
if not exist "%DIR%" exit /b 0

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$dir = '%DIR%';" ^
  "for($i=0;$i -lt 5;$i++){" ^
  "  if(Test-Path -LiteralPath $dir){" ^
  "    try { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop; break }" ^
  "    catch { Start-Sleep -Milliseconds 500 }" ^
  "  } else { break }" ^
  "}"  >nul 2>&1

if exist "%DIR%" exit /b 1
exit /b 0


rem ===================================================================
rem Subroutine: build_one  (arg1 = script filename, e.g. "data_import.py")
rem ===================================================================
:build_one
setlocal
set "FN=%~1"
if not defined FN (
  echo   [SKIP] empty filename
  endlocal & goto :eof
)
if not exist "%SCRIPT_PATH%\%FN%" (
  echo   [SKIP] %FN% not found
  endlocal & goto :eof
)
echo   -> %FN%
pyinstaller --noconfirm --clean --onefile --windowed ^
  --distpath "%TOOLS_STAGE_DIR%" ^
  --workpath "%BUILD_FOLDER_ROOT%\_tmp_%~n1" ^
  "%SCRIPT_PATH%\%FN%"
if errorlevel 1 echo      [XX] ERROR compiling %FN%
endlocal & goto :eof
