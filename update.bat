@echo off
chcp 65001 > nul
title Alt-RAG Updater

echo ===================================================
echo     Обновление llama.cpp до актуальной версии
echo ===================================================
echo.
echo ВНИМАНИЕ: Перед началом убедитесь, что приложение 
echo и сервер llama.cpp полностью закрыты, иначе файлы 
echo не смогут перезаписаться.
echo.
pause

echo.
echo Запуск скрипта обновления...
python scripts\update_llamacpp.py

echo.
pause