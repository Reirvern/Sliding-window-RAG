@echo off
:: Фикс кодировки для кириллицы (кракозябры убраны)
chcp 65001 > nul

title Alt-RAG Gradio WebUI
call .\rag_venv\Scripts\activate.bat

echo Проверка установки Gradio...
pip show gradio > nul 2>&1
if %errorlevel% neq 0 (
    echo Установка Gradio...
    pip install gradio
)

echo Запуск Web сервера...
python interface\webui.py

pause