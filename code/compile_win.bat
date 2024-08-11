@echo off

echo Compiling Python scripts...

setlocal

echo Setting up paths etc.

:: Get the full path to the batch script
set "SCRIPT_PATH=%~dp0"

:: Remove the trailing backslash for correct path manipulation
set "SCRIPT_PATH=%SCRIPT_PATH:~0,-1%"

:: Get the path of the parent directory
for %%i in ("%SCRIPT_PATH%") do set "PARENT_DIR=%%~dpi"

:: Echo the values for verification
echo SCRIPT_PATH is %SCRIPT_PATH%
echo PARENT_DIR is %PARENT_DIR%

:: Define the paths for build and dist folders
set "BUILD_FOLDER_ROOT=%PARENT_DIR%build"
set "BUILD_FOLDER_SYSTEM=%BUILD_FOLDER_ROOT%\system"
set "DIST_FOLDER_ROOT=%PARENT_DIR%dist"
set "DIST_FOLDER_SYSTEM=%DIST_FOLDER_ROOT%\system"
set "SCRIPT_FOLDER_SYSTEM=%SCRIPT_PATH%\system"
set "SYSTEM_RESOURCES_PATH=%SCRIPT_PATH%\system_resources"

:: Echo the values for verification
echo BUILD_FOLDER_ROOT is %BUILD_FOLDER_ROOT%
echo BUILD_FOLDER_SYSTEM is %BUILD_FOLDER_SYSTEM%
echo DIST_FOLDER_ROOT is %DIST_FOLDER_ROOT%
echo DIST_FOLDER_SYSTEM is %DIST_FOLDER_SYSTEM%
echo SCRIPT_FOLDER_SYSTEM is %SCRIPT_FOLDER_SYSTEM%
echo SYSTEM_RESOURCES_PATH is %SYSTEM_RESOURCES_PATH%

:: Ensure the target directories exist
if not exist "%DIST_FOLDER_ROOT%" (
    mkdir "%DIST_FOLDER_ROOT%"
)
if not exist "%DIST_FOLDER_SYSTEM%" (
    mkdir "%DIST_FOLDER_SYSTEM%"
)

echo Fetching config.ini...
xcopy "%SCRIPT_FOLDER_SYSTEM%\config.ini" "%DIST_FOLDER_SYSTEM%" /Y
if errorlevel 1 echo Error copying config.ini

echo Copying folders...
xcopy "%SYSTEM_RESOURCES_PATH%" "%DIST_FOLDER_ROOT%\system_resources" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying system_resources
xcopy "%SCRIPT_PATH%\input" "%DIST_FOLDER_ROOT%\input" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying input
xcopy "%SCRIPT_PATH%\output" "%DIST_FOLDER_ROOT%\output" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying output
xcopy "%SCRIPT_PATH%\qgis" "%DIST_FOLDER_ROOT%\qgis" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying qgis
xcopy "%SCRIPT_PATH%\docs" "%DIST_FOLDER_ROOT%\docs" /E /I /Y >nul 2>&1
if errorlevel 1 echo Error copying docs

echo Working in this folder: %BUILD_FOLDER_ROOT%
echo Distribution folder will be: %DIST_FOLDER_ROOT%

:: Activate the virtual environment
call "%PARENT_DIR%.venv\Scripts\activate.bat"

echo Activation: "%PARENT_DIR%.venv\Scripts\activate.bat"
echo Python executable: %PARENT_DIR%.venv\Scripts\python.exe

:: Start the compilation
set "PYINSTALLER_CMD=pyinstaller --onefile --windowed --icon=%SYSTEM_RESOURCES_PATH%\mesa.ico --collect-all ttkbootstrap --collect-all tkinterweb --hidden-import=ttkbootstrap --distpath=%DIST_FOLDER_SYSTEM% --workpath=%BUILD_FOLDER_SYSTEM% --add-data %SCRIPT_FOLDER_SYSTEM%;system --add-data %SYSTEM_RESOURCES_PATH%;system_resources"

echo pyinstaller cmd: %PYINSTALLER_CMD%

:: Compile Python scripts in the system folder
for %%f in (
    01_import.py,
    04_edit_asset_group.py,
    04_edit_geocode_group.py,
    04_edit_input.py,
    06_process.py,
    07_edit_atlas.py,
    07_make_atlas.py,
    08_admin_lines.py,
    08_edit_lines.py
) do (
    echo Working on %%f in %SCRIPT_FOLDER_SYSTEM%
    if exist "%SCRIPT_FOLDER_SYSTEM%\%%f" (
        %PYINSTALLER_CMD% "%SCRIPT_FOLDER_SYSTEM%\%%f" >nul 2>&1
        if errorlevel 1 echo Error compiling %%f
    ) else (
        echo File %%f does not exist in %SCRIPT_FOLDER_SYSTEM%
    )
)

:: Move compiled executables back to the system source
for %%f in (
    01_import.exe,
    04_edit_asset_group.exe,
    04_edit_geocode_group.exe,
    04_edit_input.exe,
    06_process.exe,
    07_edit_atlas.exe,
    07_make_atlas.exe,
    08_admin_lines.exe,
    08_edit_lines.exe
) do (
    if exist "%DIST_FOLDER_SYSTEM%\%%f" (
        move "%DIST_FOLDER_SYSTEM%\%%f" "%SCRIPT_FOLDER_SYSTEM%"
    ) else (
        echo Error: %%f was not created
    )
)

:: Compile the mesa.py script
echo Working on mesa.py in %SCRIPT_FOLDER_ROOT%
%PYINSTALLER_CMD% "%SCRIPT_FOLDER_ROOT%\mesa.py"
if errorlevel 1 echo Error compiling mesa.py

echo Compilation complete. You will find the compiled code here: %DIST_FOLDER_ROOT%

:: Clean up build folders and .spec files
if exist "%BUILD_FOLDER_ROOT%" (
    echo Deleting the work folder: %BUILD_FOLDER_ROOT%
    rmdir /s /q "%BUILD_FOLDER_ROOT%"
    echo Folder deleted successfully.
) else (
    echo Work folder does not exist: %BUILD_FOLDER_ROOT%
)

:: Just delete the .spec files
del "%SCRIPT_FOLDER_ROOT%\*.spec" /q
if errorlevel 1 echo Error deleting spec files
echo All .spec files deleted

echo Compilation completed
