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
set "BUILD_FOLDER_ROOT=%PARENT_DIR%build"
set "BUILD_FOLDER_SYSTEM=%PARENT_DIR%build\system"

:: Define the dist folder path
set "DIST_FOLDER_ROOT=%PARENT_DIR%dist"
set "DIST_FOLDER_SYSTEM=%PARENT_DIR%dist\system"

:: Define the system folder
set "INPUT_FOLDER_ROOT=%PARENT_DIR%"
set "INPUT_FOLDER_SYSTEM=%PARENT_DIR%system"

:: Ensure the target directory exists
if not exist "%DIST_FOLDER_SYSTEM%" (
    mkdir "%DIST_FOLDER_SYSTEM%"
)

echo
echo Fetching config.ini...
xcopy "%SCRIPT_PATH%\system\config.ini" "%DIST_FOLDER_SYSTEM%" /Y
if errorlevel 1 echo Error copying config.ini

echo
echo Copying folders...
echo -system_resources...
xcopy "%SCRIPT_PATH%\system_resources" "%DIST_FOLDER_ROOT%\system_resources" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying system_resources

echo -input...
xcopy "%SCRIPT_PATH%\input" "%DIST_FOLDER_ROOT%\input" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying input

echo -output...
xcopy "%SCRIPT_PATH%\output" "%DIST_FOLDER_ROOT%\output" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying output

echo -qgis...
xcopy "%SCRIPT_PATH%\qgis" "%DIST_FOLDER_ROOT%\qgis" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying qgis

echo -docs...
xcopy "%SCRIPT_PATH%\docs" "%DIST_FOLDER_ROOT%\docs" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying docs

echo
echo Working in this folder: %BUILD_FOLDER_ROOT%

echo Distribution folder will be: %DIST_FOLDER_ROOT%

:: Start the compilation
set "PYINSTALLER_CMD=pyinstaller --onefile --collect-all ttkbootstrap --collect-all tkinterweb --hidden-import=ttkbootstrap --distpath=%DIST_FOLDER_ROOT% --workpath=%BUILD_FOLDER_ROOT%"

echo Working on mesa.py
%PYINSTALLER_CMD% mesa.py >nul 2>&1
if errorlevel 1 echo Error compiling mesa.py

echo Working on 01_import.py
%PYINSTALLER_CMD% 01_import.py >nul 2>&1
if errorlevel 1 echo Error compiling 01_import.py

echo Working on 02_present_files.py
%PYINSTALLER_CMD% 02_present_files.py >nul 2>&1
if errorlevel 1 echo Error compiling 02_present_files.py

echo Working on 04_edit_asset_group.py
%PYINSTALLER_CMD% 04_edit_asset_group.py >nul 2>&1
if errorlevel 1 echo Error compiling 04_edit_asset_group.py

echo Working on 04_edit_geocode_group.py
%PYINSTALLER_CMD% 04_edit_geocode_group.py >nul 2>&1
if errorlevel 1 echo Error compiling 04_edit_geocode_group.py

echo Working on 04_edit_input.py
%PYINSTALLER_CMD% 04_edit_input.py >nul 2>&1
if errorlevel 1 echo Error compiling 04_edit_input.py

echo Working on 06_process.py
%PYINSTALLER_CMD% 06_process.py >nul 2>&1
if errorlevel 1 echo Error compiling 06_process.py

echo Working on 07_edit_atlas.py
%PYINSTALLER_CMD% 07_edit_atlas.py >nul 2>&1
if errorlevel 1 echo Error compiling 07_edit_atlas.py

echo Working on 07_make_atlas.py
%PYINSTALLER_CMD% 07_make_atlas.py >nul 2>&1
if errorlevel 1 echo Error compiling 07_make_atlas.py

echo Working on 08_admin_lines.py
%PYINSTALLER_CMD% 08_admin_lines.py >nul 2>&1
if errorlevel 1 echo Error compiling 08_admin_lines.py

echo Working on 08_edit_lines.py
%PYINSTALLER_CMD% 08_edit_lines.py >nul 2>&1
if errorlevel 1 echo Error compiling 08_edit_lines.py

echo Compilation complete. You will find the compiled code here: %DIST_FOLDER_ROOT%

echo

:: Clean up build folders and .spec files

:: Check if the folder exists
if exist "%BUILD_FOLDER_ROOT%" (
    echo Deleting the work folder: %BUILD_FOLDER_ROOT%
    rmdir /s /q "%BUILD_FOLDER_ROOT%"
    echo Folder deleted successfully.
) else (
    echo Work folder does not exist: %BUILD_FOLDER_ROOT%
)

echo Build folders deleted

echo

:: Just delete the -spec-files
del "*.spec" /q

echo All .spec-files deleted
