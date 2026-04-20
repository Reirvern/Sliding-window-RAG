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

REM 1. ПРОВЕРКА И ЛОКАЛЬНАЯ УСТАНОВКА PYTHON 3.11
if exist "%PYTHON_EXE%" goto :skip_python

echo [1/5] Скачивание полноценного Python 3.11.9...
curl.exe -L -o python_installer.exe "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

echo.
echo [2/5] Локальная установка Python, Tkinter и pip...
echo (Откроется окно установки, ничего нажимать не нужно)
start /wait python_installer.exe /passive InstallAllUsers=0 Include_launcher=0 Include_tcltk=1 Include_pip=1 TargetDir="%PYTHON_DIR%"
del python_installer.exe

echo Python успешно установлен!
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