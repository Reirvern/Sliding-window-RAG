@echo off
setlocal

rem Устанавливаем кодировку консоли на UTF-8 для корректного отображения вывода
chcp 65001 > nul

echo.
echo === Llama.cpp Python Installer ===
echo.

rem --- Шаг 1: Создание чистого виртуального окружения ---
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

rem --- Шаг 2: Установка базовых зависимостей ---
echo Installing basic requirements from requirements.txt...
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
echo --- Installing with GPU Support (Pre-compiled) ---
echo.
echo Please select your installed CUDA Toolkit version.
echo The installer will try to download a pre-compiled package.
echo.
echo   1. CUDA 12.1 (Recommended for RTX 30xx/40xx series)
echo   2. CUDA 11.8
echo   3. CUDA 11.7
echo.
set /p CUDA_VERSION_CHOICE="Enter the number of your version (1, 2, or 3): "

set "EXTRA_INDEX_URL=https://huggingface.github.io/llama-cpp-python/whl/cu"

if "%CUDA_VERSION_CHOICE%"=="1" set "CUDA_TAG=121"
if "%CUDA_VERSION_CHOICE%"=="2" set "CUDA_TAG=118"
if "%CUDA_VERSION_CHOICE%"=="3" set "CUDA_TAG=117"

if not defined CUDA_TAG (
    echo Invalid selection. Please run the script again.
    pause
    exit /b 1
)

echo Uninstalling any existing llama-cpp-python to ensure a clean install...
pip uninstall -y llama-cpp-python

echo.
echo Installing llama-cpp-python for CUDA %CUDA_TAG%...
echo This will download the pre-compiled binary.
echo.

rem Установка с указанием версии CUDA
pip install llama-cpp-python --force-reinstall --no-cache-dir --index-url https://pypi.org/simple/ --extra-index-url %EXTRA_INDEX_URL%%CUDA_TAG%

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install llama-cpp-python with CUDA %CUDA_TAG% support.
    echo This might mean a pre-compiled version for your Python/CUDA combination was not found.
    echo You may need to compile from source.
    echo.
    pause
) else (
    echo.
    echo Successfully installed llama-cpp-python with CUDA %CUDA_TAG% support!
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
