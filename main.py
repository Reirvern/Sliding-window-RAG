# main.py
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
from core.utils.config_loader import load_config # Используем load_config для общих настроек приложения
from core.utils.localization.translator import Translator

# Фабрики
from interface.factory import create_interface
from core.factories.engine_factory import create_rag_engine
from core.utils.config_loader import load_rag_config # Используем load_rag_config для RAG-специфичных настроек

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
    # Загрузка общей конфигурации приложения
    app_config = load_config(config_path)

    # Настройка логгера
    logger = setup_logger(
        log_level=app_config['logging']['level'],
        log_to_console=app_config['logging']['log_to_console'],
        console_log_level=app_config['logging'].get('console_level', 'INFO'), # НОВОЕ: Получаем console_level
        log_to_file=app_config['logging']['log_to_file'],
        log_file_path=Path(app_config['logging']['log_file_path'])
    )
    
    # Загрузка RAG конфигурации (отдельно, так как она более сложная)
    rag_config = load_rag_config(Path("configs/rag_engine_config.json")) # Предполагаем фиксированный путь для RAG конфига

    # Инициализация переводчика
    translator = Translator(language=app_config['language'])

    return {
        "app_config": app_config,
        "rag_config": rag_config,
        "logger": logger,
        "translator": translator
    }

if __name__ == "__main__":
    config_file_path = PROJECT_ROOT / "configs" / "config.json"
    
    # Устанавливаем обработчик для необработанных исключений
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: log_unhandled_exception(global_logger, exc_type, exc_value, exc_traceback)

    try:
        # Инициализация системы
        system_components = initialize_system(config_file_path)
        app_config = system_components["app_config"]
        rag_config = system_components["rag_config"]
        logger = system_components["logger"]
        translator = system_components["translator"]

        global_logger = logger
        global_config = app_config # Сохраняем app_config в global_config
        global_translator = translator
        
        logger.info("=" * 60)
        logger.info(translator.translate("app_start"))
        logger.debug(f"Версия Python: {sys.version}")
        logger.info(translator.translate("current_language").format(language=app_config['language']))
        logger.info("=" * 60)
        
        # Создание RAG Engine
        rag_engine = create_rag_engine(
            rag_config=rag_config, # ИЗМЕНЕНО: Имя параметра изменено с 'config' на 'rag_config'
            logger=logger,
            translator=translator
        )
        
        # Создание интерфейса
        app_interface = create_interface(
            interface_type=app_config['interface'],
            config=app_config, # Передаем app_config здесь
            rag_engine=rag_engine,
            logger=logger,
            translator=translator
        )
        
        # Регистрация интерфейса как наблюдателя для RAG Engine
        rag_engine.add_observer(app_interface)

        # Запуск основного цикла приложения
        logger.info(translator.translate("interface_start"))
        app_interface.run()
        
        logger.info(translator.translate("app_shutdown_success"))
        

    except Exception as e:
        import traceback
        traceback.print_exc()
        if global_logger:
            global_logger.critical(translator.translate("critical_error_main_thread"))
            log_unhandled_exception(global_logger, e)
        else:
            # Fallback if logger is not initialized
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Необработанное исключение в основном потоке: {e}")
            traceback.print_exc()
