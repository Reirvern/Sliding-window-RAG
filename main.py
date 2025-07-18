import sys
import io
import json
import logging
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent

if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
# Утилиты
from core.utils.logger import setup_logger
from core.utils.error_handling import log_unhandled_exception
from core.utils.config_loader import load_config, load_rag_config
from core.utils.localization.translator import Translator

# Фабрики
from interface.factory import create_interface
from core.factories.engine_factory import create_rag_engine
from core.domain.models import RAGConfig

# Инициализация глобальных переменных для доступа к основным компонентам
global_logger = None
global_config = None
global_translator = None

def initialize_system(config_path: Path) -> Dict[str, Any]:
    """
    Инициализирует основные компоненты системы
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Кортеж с основными компонентами (config, logger, translator)
    """
    # Загрузка конфигурации
    config = load_config(config_path)
    
    # Настройка логгера
    # ИЗМЕНЕНО: Передаем строковые значения уровней логирования
    logger = setup_logger(
        log_level=config['logging']['level'], # Передаем строку
        log_to_console=config['logging']['log_to_console'],
        console_log_level=config['logging']['console_level'], # Передаем строку
        log_to_file=config['logging']['log_to_file'],
        log_file_path=Path('logs') / 'app.log'
    )
    
    # Инициализация системы локализации
    translator = Translator(config['language'])
    
    return config, logger, translator

def main():
    """Основная функция запуска приложения"""
    try:
        # Определение путей
        BASE_DIR = Path(__file__).parent
        CONFIG_PATH = BASE_DIR / 'configs' / 'config.json'
        RAG_CONFIG_PATH = BASE_DIR / 'configs' / 'rag_engine_config.json' 
        
        # Инициализация системы
        config, logger, translator = initialize_system(CONFIG_PATH)
        
        # Сохраняем глобально для обработки ошибок
        global global_logger, global_config, global_translator
        global_logger = logger
        global_config = config
        global_translator = translator
        
        logger.info("=" * 60)
        logger.info("Запуск Alt-RAG системы")
        logger.debug(f"Версия Python: {sys.version}")
        logger.info(f"Текущий язык интерфейса: {config['language']}")
        logger.info("=" * 60)
        
        # Загрузка RAG конфига
        rag_config: RAGConfig = load_rag_config(RAG_CONFIG_PATH)

        # Создание RAG Engine
        rag_engine = create_rag_engine(
            rag_config=rag_config,
            logger=logger,
            translator=translator
        )
        
        # Создание интерфейса
        app_interface = create_interface(
            interface_type=config['interface'],
            config=config,
            rag_engine=rag_engine,
            logger=logger,
            translator=translator
        )
        

        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ: РЕГИСТРАЦИЯ CLIInterface КАК НАБЛЮДАТЕЛЯ RAGEngine ---
        # Поскольку CLIInterface является Observer, и RAGEngine является Observable,
        # CLIInterface должен быть добавлен как наблюдатель к RAGEngine,
        # чтобы получать уведомления о прогрессе.
        rag_engine.add_observer(app_interface)
        # ----------------------------------------------------------------------------------

        # Запуск основного цикла приложения
        logger.info("Запуск основного интерфейса...")
        app_interface.run()
        
        logger.info("Приложение завершило работу успешно")
        

    except Exception as e:
        import traceback
        traceback.print_exc()
        if global_logger:
            global_logger.critical("КРИТИЧЕСКАЯ ОШИБКА: Необработанное исключение в основном потоке")
            log_unhandled_exception(global_logger, e)
        else:
            print(f"Critical error before logger initialization: {str(e)}")

if __name__ == "__main__":
    # Точка входа при запуске скрипта
    main()
