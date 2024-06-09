:: Compilation script.
:: Consider using UPX-co-compilation package for smaller files.

@echo off

echo Compiling Python scripts...

setlocal

echo
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

:: Path of the work folder relative to the script
set "BUILD_FOLDER_ROOT=%PARENT_DIR%build"
set "BUILD_FOLDER_SYSTEM=%PARENT_DIR%build\system"

:: Define the dist folder path
set "DIST_FOLDER_ROOT=%PARENT_DIR%dist"
set "DIST_FOLDER_SYSTEM=%PARENT_DIR%dist\system"

:: Define the system folder path
set "SCRIPT_FOLDER_ROOT=%SCRIPT_PATH%"
set "SCRIPT_FOLDER_SYSTEM=%SCRIPT_PATH%\system"

:: Echo the values for verification
echo BUILD_FOLDER_ROOT is %BUILD_FOLDER_ROOT%
echo BUILD_FOLDER_SYSTEM is %BUILD_FOLDER_SYSTEM%
echo DIST_FOLDER_ROOT is %DIST_FOLDER_ROOT%
echo DIST_FOLDER_SYSTEM is %DIST_FOLDER_SYSTEM%
echo SCRIPT_FOLDER_ROOT is %SCRIPT_FOLDER_ROOT%
echo SCRIPT_FOLDER_SYSTEM is %SCRIPT_FOLDER_SYSTEM%

:: Ensure the target directories exist
if not exist "%DIST_FOLDER_ROOT%" (
    mkdir "%DIST_FOLDER_ROOT%"
)
if not exist "%DIST_FOLDER_SYSTEM%" (
    mkdir "%DIST_FOLDER_SYSTEM%"
)

echo
echo Fetching config.ini...
xcopy "%SCRIPT_FOLDER_SYSTEM%\config.ini" "%DIST_FOLDER_SYSTEM%" /Y
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
echo

:: Start the compilation
set "PYINSTALLER_CMD_ROOT=pyinstaller --onefile --windowed --icon=system_resources\mesa.ico --collect-all ttkbootstrap --collect-all tkinterweb --hidden-import=ttkbootstrap --distpath=%DIST_FOLDER_ROOT% --workpath=%BUILD_FOLDER_ROOT% --add-data %SCRIPT_FOLDER_SYSTEM%;system --add-data %SCRIPT_PATH%\system_resources;system_resources"
set "PYINSTALLER_CMD_SYSTEM=pyinstaller --onefile --windowed --collect-all ttkbootstrap --collect-all tkinterweb --hidden-import=ttkbootstrap --distpath=%DIST_FOLDER_SYSTEM% --workpath=%BUILD_FOLDER_SYSTEM% --add-data %SCRIPT_FOLDER_SYSTEM%;system --add-data %SCRIPT_PATH%\system_resources;system_resources"

echo Working on 01_import.py in %SCRIPT_FOLDER_SYSTEM%
if exist "%SCRIPT_FOLDER_SYSTEM%\01_import.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\01_import.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\01_import.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 01_import.py
) else (
    echo File 01_import.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 04_edit_asset_group.py in %SCRIPT_FOLDER_SYSTEM%
if exist "%SCRIPT_FOLDER_SYSTEM%\04_edit_asset_group.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\04_edit_asset_group.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\04_edit_asset_group.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 04_edit_asset_group.py
) else (
    echo File 04_edit_asset_group.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 04_edit_geocode_group.py in %SCRIPT_FOLDER_SYSTEM%
if exist "%SCRIPT_FOLDER_SYSTEM%\04_edit_geocode_group.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\04_edit_geocode_group.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\04_edit_geocode_group.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 04_edit_geocode_group.py
) else (
    echo File 04_edit_geocode_group.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 04_edit_input.py in %SCRIPT_FOLDER_SYSTEM%
if exist "%SCRIPT_FOLDER_SYSTEM%\04_edit_input.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\04_edit_input.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\04_edit_input.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 04_edit_input.py
) else (
    echo File 04_edit_input.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 06_process.py
if exist "%SCRIPT_FOLDER_SYSTEM%\06_process.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\06_process.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\06_process.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 06_process.py
) else (
    echo File 06_process.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 07_edit_atlas.py
if exist "%SCRIPT_FOLDER_SYSTEM%\07_edit_atlas.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\07_edit_atlas.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\07_edit_atlas.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 07_edit_atlas.py
) else (
    echo File 07_edit_atlas.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 07_make_atlas.py
if exist "%SCRIPT_FOLDER_SYSTEM%\07_make_atlas.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\07_make_atlas.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\07_make_atlas.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 07_make_atlas.py
) else (
    echo File 07_make_atlas.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 08_admin_lines.py
if exist "%SCRIPT_FOLDER_SYSTEM%\08_admin_lines.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\08_admin_lines.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\08_admin_lines.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 08_admin_lines.py
) else (
    echo File 08_admin_lines.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo Working on 08_edit_lines.py
if exist "%SCRIPT_FOLDER_SYSTEM%\08_edit_lines.py" (
    echo %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\08_edit_lines.py"
    %PYINSTALLER_CMD_SYSTEM% "%SCRIPT_FOLDER_SYSTEM%\08_edit_lines.py" >nul 2>&1
    if errorlevel 1 echo Error compiling 08_edit_lines.py
) else (
    echo File 08_edit_lines.py does not exist in %SCRIPT_FOLDER_SYSTEM%
)

echo -01_import.exe back in system source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\01_import.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 01_import
echo

echo -04_edit_asset_group.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\04_edit_asset_group.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 04_edit_asset_group
echo

echo -04_edit_geocode_group.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\04_edit_geocode_group.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 04_edit_geocode_group
echo

echo -04_edit_input.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\04_edit_input.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 04_edit_input
echo

echo -06_process.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\06_process.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 06_process
echo

echo -07_edit_atlas.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\07_edit_atlas.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 07_edit_atlas
echo

echo -07_make_atlas.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\07_make_atlas.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 07_make_atlas
echo

echo -08_admin_lines.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\08_admin_lines.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 08_admin_lines
echo

echo -08_edit_lines.exe back in system at source for inclusion in mesa.exe.
move  "%DIST_FOLDER_ROOT%\system\08_edit_lines.exe" "%SCRIPT_PATH%\system"
if errorlevel 1 echo Error copying 08_edit_lines
echo

echo Working on mesa.py in %SCRIPT_FOLDER_ROOT%
echo %PYINSTALLER_CMD_ROOT% "%SCRIPT_FOLDER_ROOT%\mesa.py"
%PYINSTALLER_CMD_ROOT% "%SCRIPT_FOLDER_ROOT%\mesa.py" >nul 2>&1
if errorlevel 1 echo Error compiling mesa.py

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


:: Delete all .exe files in the original code folder
echo Deleting all .exe files in the original code folder: %SCRIPT_FOLDER_ROOT%
del "%SCRIPT_FOLDER_ROOT%\system\*.exe" /q
if errorlevel 1 echo Error deleting .exe files
echo All .exe files deleted
echo


:: Just delete the -spec-files
del "*.spec" /q

echo All .spec-files deleted
echo
echo Compilation completed

