@echo off
python -m venv rag_venv
call .\rag_venv\Scripts\activate.bat
pip install -r requirements.txt
echo installing successfull!
pause
