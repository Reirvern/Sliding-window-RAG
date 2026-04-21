@echo off
:: Фикс кодировки
chcp 65001 > nul
title Alt-RAG Portable Installer

echo ===================================================
echo     Установка Alt-RAG (Portable версия)
echo ===================================================
echo.

REM Убиваем зависшие процессы
taskkill /F /IM llama-server.exe > nul 2>&1
set PYTHONNOUSERSITE=1

REM Папки проекта
set "PYTHON_DIR=%~dp0python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"

REM 1. ПРОВЕРКА И ЛОКАЛЬНАЯ УСТАНОВКА PYTHON
if exist "%PYTHON_EXE%" goto :skip_python

echo [1/5] Скачивание полноценного Python 3.10.11 (Portable Zip)...
curl.exe -L -o python.zip "https://www.nuget.org/api/v2/package/python/3.10.11"

echo.
echo [2/5] Распаковка Python и настройка окружения...
powershell -Command "Expand-Archive -Path 'python.zip' -DestinationPath 'python_temp' -Force"

REM Перемещаем ядро Питона в нашу папку python
move python_temp\tools "%PYTHON_DIR%" > nul

REM Удаляем мусор
rd /s /q python_temp
del python.zip

echo [~] Скачивание и установка pip...
curl.exe -L -o get-pip.py "https://bootstrap.pypa.io/get-pip.py"
"%PYTHON_EXE%" get-pip.py
del get-pip.py

echo.
echo Python успешно распакован и настроен!
goto :install_deps

:skip_python
echo [1/5] Портативный Python уже установлен. Пропуск...
echo [2/5] Установка не требуется.

:install_deps
echo.
echo [3/5] Установка зависимостей проекта...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo.
echo [4/5] Создание рабочих директорий...
if not exist "runtimes" mkdir runtimes
if not exist "runtimes\cpu_x64" mkdir runtimes\cpu_x64
if not exist "runtimes\cuda12" mkdir runtimes\cuda12
if not exist "runtimes\cuda13" mkdir runtimes\cuda13
if not exist "runtimes\vulkan" mkdir runtimes\vulkan

if not exist "models" mkdir models
if not exist "user_data" mkdir user_data
if not exist "user_data\outputs" mkdir user_data\outputs
if not exist "user_data\temp_uploads" mkdir user_data\temp_uploads

echo.
echo [5/5] Скачивание актуальных ядер llama.cpp...
"%PYTHON_EXE%" scripts\update_llamacpp.py

echo.
echo ===================================================
echo Установка полностью завершена! 
echo Теперь вы можете запускать run_gui.bat или run_webui.bat
echo ===================================================
pause