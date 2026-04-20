@echo off
chcp 65001 > nul
title Alt-RAG Gradio WebUI

:: Полная изоляция от глобальных библиотек Windows
set PYTHONNOUSERSITE=1

echo Запуск Web сервера...
.\python\python.exe interface\webui.py

pause