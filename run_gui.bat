@echo off
title Alt-RAG Studio
call .\rag_venv\Scripts\activate.bat

:: Запускаем main.py с аргументом --gui
python main.py --gui

pause