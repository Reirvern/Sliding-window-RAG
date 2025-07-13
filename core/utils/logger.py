import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

def setup_logger(log_level: str, 
                log_to_console: bool, 
                log_to_file: bool,
                log_file_path: Path) -> logging.Logger:
    """
    Инициализирует и настраивает логгер системы
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Флаг вывода в консоль
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
    log_level_numeric = level_mapping.get(log_level.upper(), logging.DEBUG)
    logger.setLevel(log_level_numeric)
    
    # Форматтер с таймстампом, уровнем и сообщением
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Очистка предыдущих обработчиков
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Консольный обработчик
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level_numeric)
        logger.addHandler(console_handler)
    
    # Файловый обработчик с ротацией
    if log_to_file:
        # Создаем директорию для логов, если не существует
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ротация по размеру (15 МБ) + очистка старых логов
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=15 * 1024 * 1024,  # 15 МБ
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level_numeric)
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
            
            # Удаляем файлы старше cutoff
            if mtime < cutoff:
                try:
                    log_file.unlink()
                    print(f"Удален старый лог: {log_file.name}")
                except Exception as e:
                    print(f"Ошибка удаления лога {log_file.name}: {str(e)}")