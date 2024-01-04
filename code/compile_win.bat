@echo off
echo Compiling Python scripts...

setlocal

:: Get the full path to the batch script
set "SCRIPT_PATH=%~dp0"

:: Remove the trailing backslash for correct path manipulation
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

:: Get the path of the parent directory
for %%i in ("%SCRIPT_PATH%") do set "PARENT_DIR=%%~dpi"

:: Path of the work folder relative to the script
set "BUILD_FOLDER=%PARENT_DIR%build"

:: Define the dist folder path
set "DIST_FOLDER=%PARENT_DIR%dist"

echo Working in this folder: %BUILD_FOLDER%

echo Distribution folder will be: %DIST_FOLDER%

:: Start the compilation
echo Working on user_interface.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" user_interface.py >nul 2>&1

echo Working on 01_import.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 01_import.py >nul 2>&1

echo Working on 02_present_files.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 02_present_files.py >nul 2>&1

echo Working on 04_edit_asset_group.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 04_edit_asset_group.py >nul 2>&1

echo Working on 04_edit_geocode_group.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 04_edit_geocode_group.py >nul 2>&1

echo Working on 04_edit_input.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 04_edit_input.py >nul 2>&1

echo Working on 06_process.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 06_process.py >nul 2>&1

echo Working on 07_edit_atlas.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 07_edit_atlas.py >nul 2>&1

echo Working on 07_make_atlas.py
pyinstaller --onefile --distpath="%DIST_FOLDER%" --workpath="%BUILD_FOLDER%" 07_make_atlas.py >nul 2>&1

echo Compilation complete. You will finde the compiled code herE: %DIST_FOLDER%

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

::Just delete the -spec-files
del "*.spec" /q

echo All .spec-files deleted

pause