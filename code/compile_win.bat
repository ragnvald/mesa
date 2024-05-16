:: Compilation script.
:: Consider using UPX-co-compilation package for smaller files.

echo Compiling Python scripts...
echo off

setlocal

echo
echo Setting up paths etc.

:: Get the full path to the batch script
set "SCRIPT_PATH=%~dp0"

:: Remove the trailing backslash for correct path manipulation
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

:: Get the path of the parent directory
for %%i in ("%SCRIPT_PATH%") do set "PARENT_DIR=%%~dpi"

:: Path of the work folder relative to the script
set "BUILD_FOLDER_ROOT      = %PARENT_DIR%build"
set "BUILD_FOLDER_SYSTEM    = %PARENT_DIR%build/system"

:: Define the dist folder path
set "DIST_FOLDER_ROOT       = %PARENT_DIR%dist"
set "DIST_FOLDER_SYSTEM     = %PARENT_DIR%dist/system"

:: Define the system folder
set "INPUT_FOLDER_ROOT    = %PARENT_DIR%"
set "INPUT_FOLDER_SYSTEM    = %PARENT_DIR%system"

echo
echo Fetching config.ini...
xcopy "%SCRIPT_PATH%\config.ini" "%DIST_FOLDER%" /Y >nul 2>&1

echo
echo Copying folders...
echo -system_resources...
xcopy "%SCRIPT_PATH%\system_resources" "%DIST_FOLDER%\system_resources" /E /I /Y >nul 2>&1

echo -input...
xcopy "%SCRIPT_PATH%\input" "%DIST_FOLDER%\input" /E /I /Y >nul 2>&1

echo -output...
xcopy "%SCRIPT_PATH%\output" "%DIST_FOLDER%\output" /E /I /Y >nul 2>&1

echo -qgis...
xcopy "%SCRIPT_PATH%\qgis" "%DIST_FOLDER%\qgis" /E /I /Y >nul 2>&1

echo -docs...
xcopy "%SCRIPT_PATH%\docs" "%DIST_FOLDER%\docs" /E /I /Y >nul 2>&1

echo
echo Working in this folder: %BUILD_FOLDER%

echo Distribution folder will be: %DIST_FOLDER%

:: Start the compilation

echo Working on mesa.py
pyinstaller --onefile --collect-all ttkbootstrap --collect-all tkinterweb --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_ROOT%" --workpath="%BUILD_FOLDER_ROOT%" user_interface.py >nul 2>&1

echo Working on 01_import.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 01_import.py >nul 2>&1

echo Working on 02_present_files.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 02_present_files.py >nul 2>&1

echo Working on 04_edit_asset_group.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 04_edit_asset_group.py >nul 2>&1

echo Working on 04_edit_geocode_group.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 04_edit_geocode_group.py >nul 2>&1

echo Working on 04_edit_input.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 04_edit_input.py >nul 2>&1

echo Working on 06_process.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 06_process.py >nul 2>&1

echo Working on 07_edit_atlas.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 07_edit_atlas.py >nul 2>&1

echo Working on 07_make_atlas.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER_SYSTEM%" 07_make_atlas.py >nul 2>&1

echo Working on 08_admin_lines.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 08_admin_lines.py >nul 2>&1

echo Working on 08_edit_lines.py
pyinstaller --onefile --collect-all ttkbootstrap --hidden-import=ttkbootstrap --distpath="%DIST_FOLDER_SYSTEM%" --workpath="%BUILD_FOLDER_SYSTEM%" 08_edit_lines.py >nul 2>&1

echo Compilation complete. You will finde the compiled code here: %DIST_FOLDER_ROOT%


echo

:: Clean up build folders and .spec files

:: Check if the folder exists
if exist "%BUILD_FOLDER%" (
    echo Deleting the work folder: %BUILD_FOLDER%
    rmdir /s /q "%BUILD_FOLDER%"
    echo Folder deleted successfully.
) else (
    echo Work folder does not exist: %BUILD_FOLDER%
)

 echo Build folders deleted

echo

:: Just delete the -spec-files
 del "*.spec" /q

echo All .spec-files deleted

pause