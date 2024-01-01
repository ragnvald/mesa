@echo off
echo Compiling Python scripts...

REM Run PyInstaller for your script
pyinstaller --onefile user_interface.py
pyinstaller --onefile 01_import.py
pyinstaller --onefile 02_present_files.py
pyinstaller --onefile 03_data_structure.py
pyinstaller --onefile 04_edit_asset_group.py
pyinstaller --onefile 04_edit_geocode_group.py
pyinstaller --onefile 04_edit_input.py
pyinstaller --onefile 05_main_statistics.py
pyinstaller --onefile 06_process.py
pyinstaller --onefile 07_edit_atlas.py
pyinstaller --onefile 07_make_atlas.py

echo Compilation complete.
pause