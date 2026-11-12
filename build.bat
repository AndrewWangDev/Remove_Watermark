@echo off
echo Building Standalone Watermark Cleaner...

:: Only create dummy Tcl/Tk folders on Python 3.14+ where PyInstaller ZipFS workaround is needed
for /f "tokens=1,2 delims=." %%a in ('python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")"') do (
    if %%a EQU 3 if %%b GEQ 14 (
        if not exist "build_helpers\_tcl_data" mkdir "build_helpers\_tcl_data"
        echo. > build_helpers\_tcl_data\dummy.txt
        if not exist "build_helpers\_tk_data" mkdir "build_helpers\_tk_data"
        echo. > build_helpers\_tk_data\dummy.txt
        if not exist "build_helpers\tcl" mkdir "build_helpers\tcl"
        echo. > build_helpers\tcl\dummy.txt
        if not exist "build_helpers\tk" mkdir "build_helpers\tk"
        echo. > build_helpers\tk\dummy.txt
    )
)

:: Compile into a single standalone EXE using optimized spec
pyinstaller --noconfirm main.spec

echo Build complete! Your standalone EXE is in the dist folder.
