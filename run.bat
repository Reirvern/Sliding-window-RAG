@echo off
call .\rag_venv\Scripts\activate.bat

python main.py --config configs/default.json

pause