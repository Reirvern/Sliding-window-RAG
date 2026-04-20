@echo off
chcp 65001 > nul
title Alt-RAG CLI

:: Полная изоляция от глобальных библиотек Windows
set PYTHONNOUSERSITE=1

echo Запуск консольного интерфейса...
.\python\python.exe main.py

pause