@echo off
chcp 65001 > nul
title Alt-RAG Studio

:: Полная изоляция от глобальных библиотек Windows
set PYTHONNOUSERSITE=1

echo Запуск графического интерфейса...
.\python\python.exe main.py --gui
pause