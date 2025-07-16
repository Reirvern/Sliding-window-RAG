# core/utils/logger.py
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

def setup_logger(log_level: str, 
                log_to_console: bool, 
                console_log_level: str, # НОВЫЙ ПАРАМЕТР: Уровень логирования для консоли
                log_to_file: bool,
                log_file_path: Path) -> logging.Logger:
    """
    Инициализирует и настраивает логгер системы
    
    Args:
        log_level: Уровень логирования для общего логгера (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Флаг вывода в консоль
        console_log_level: Уровень логирования для консоли (например, INFO, WARNING, ERROR)
        log_to_file: Флаг записи в файл
        log_file_path: Путь к файлу логов
        
    Returns:
        Настроенный экземпляр логгера
    """
    logger = logging.getLogger('AltRAG')
    
    # Преобразуем строковый уровень в числовой
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Уровень для общего логгера и файлового хэндлера
    log_level_numeric = level_mapping.get(log_level.upper(), logging.DEBUG)
    logger.setLevel(log_level_numeric)
    
    # Форматтер с таймстампом, уровнем и сообщением
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Консольный хэндлер
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        # ИСПОЛЬЗУЕМ console_log_level ДЛЯ КОНСОЛЬНОГО ХЭНДЛЕРА
        console_handler_numeric_level = level_mapping.get(console_log_level.upper(), logging.INFO)
        console_handler.setLevel(console_handler_numeric_level)
        logger.addHandler(console_handler)
    
    # Файловый хэндлер
    if log_to_file:
        # Убедимся, что директория для логов существует
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ротация по размеру (15 МБ) + очистка старых логов
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=15 * 1024 * 1024,  # 15 МБ
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level_numeric) # Файловый хэндлер использует основной log_level
        logger.addHandler(file_handler)
        
        # Очистка логов старше 14 дней
        cleanup_old_logs(log_file_path.parent)
    
    return logger

def cleanup_old_logs(log_dir: Path, days_to_keep: int = 14):
    """
    Удаляет логи старше указанного количества дней
    
    Args:
        log_dir: Директория с логами
        days_to_keep: Количество дней для хранения логов
    """
    now = datetime.now()
    cutoff = now - timedelta(days=days_to_keep)
    
    for log_file in log_dir.iterdir():
        if log_file.is_file():
            # Получаем время последнего изменения
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff:
                try:
                    os.remove(log_file)
                    logging.getLogger('AltRAG').info(f"Удален старый лог-файл: {log_file}")
                except OSError as e:
                    logging.getLogger('AltRAG').error(f"Ошибка при удалении лог-файла {log_file}: {e}")

