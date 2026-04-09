@echo off
setlocal

rem Set console encoding to UTF-8 for correct output display
chcp 65001 > nul

echo.
echo === Llama.cpp Python Installer ===
echo.

rem --- Step 1: Create a clean virtual environment ---
if exist .\rag_venv (
    echo Deleting old virtual environment...
    rmdir /s /q .\rag_venv
    if errorlevel 1 (
        echo Failed to delete rag_venv. Please delete it manually and try again.
        pause
        exit /b 1
    )
    echo Virtual environment deleted.
)

echo Creating new virtual environment...
python -m venv rag_venv
if errorlevel 1 (
    echo Failed to create virtual environment. Make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

call .\rag_venv\Scripts\activate.bat

rem --- Step 2: Install basic requirements ---
echo Installing basic requirements from requirements.txt...
rem Recommendation: If llama-cpp-python is listed in requirements.txt, remove it from there,
rem to avoid installing a CPU-only version before attempting the GPU version.
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing basic requirements. Check your requirements.txt file.
    pause
    exit /b 1
)

echo Installing cmake...
pip install cmake
if errorlevel 1 (
    echo Error installing cmake.
    pause
    exit /b 1
)

:GPU_PROMPT
echo.
echo Do you want to install llama-cpp-python with GPU (CUDA) support? (Y/N)
set /p GPU_CHOICE=Enter your choice: 

if /i "%GPU_CHOICE%"=="Y" goto INSTALL_GPU
if /i "%GPU_CHOICE%"=="N" goto SKIP_GPU
echo Invalid choice. Please enter Y or N.
goto GPU_PROMPT


:INSTALL_GPU
echo.
echo --- Installing with GPU Support (Pre-compiled Wheel) ---
echo.
echo We will attempt to download and install the pre-compiled llama-cpp-python wheel for CUDA 12.1.
echo (This should work with your CUDA 12.5 installation due to CUDA's backward compatibility).
echo.
echo Please ensure you have:
echo 1. NVIDIA drivers installed for your graphics card.
echo 2. CUDA Toolkit (v12.1 or newer, like your v12.5).
echo 3. cuDNN files (bin, include, lib) copied into the corresponding folders of your CUDA Toolkit.
echo ====================================================================================
pause

rem --- Uninstall any existing llama-cpp-python to ensure a clean install ---
echo Uninstalling any existing llama-cpp-python to ensure a clean install...
pip uninstall -y llama-cpp-python

echo.
echo Attempting to download pre-compiled llama-cpp-python wheel (v0.3.13+cu121 for Python 3.10)...
set "WHEEL_FILENAME=llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl"
set "DOWNLOAD_URL=https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/%WHEEL_FILENAME%"

powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%WHEEL_FILENAME%' -UseBasicParsing"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to download the pre-compiled wheel.
    echo Please check your internet connection and if the URL is correct: %DOWNLOAD_URL%
    echo.
    pause
    goto END_SCRIPT
)
echo Successfully downloaded %WHEEL_FILENAME%.

echo.
echo Installing the downloaded wheel...
pip install "%WHEEL_FILENAME%" --force-reinstall --no-cache-dir -v

rem Delete the downloaded wheel file after installation
del "%WHEEL_FILENAME%" 2>nul

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install the downloaded llama-cpp-python wheel.
    echo This might mean the file is corrupted or there are other installation issues.
    echo.
    pause
) else (
    echo.
    echo Successfully installed llama-cpp-python with CUDA 12.1 support!
    echo.
)
goto END_SCRIPT


:SKIP_GPU
echo.
echo Skipping GPU installation. The CPU-only version from requirements.txt will be used.
echo.


:END_SCRIPT
echo Installation process complete!
pause
endlocal
