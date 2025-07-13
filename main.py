import json
import logging
from interface.factory import InterfaceFactory

def setup_logger(config: dict) -> logging.Logger:
    """
    Настраивает и возвращает логгер на основе конфигурации
    
    :param config: Словарь с настройками логгирования
    :return: Объект логгера
    """
    logger = logging.getLogger("SWRAG")
    logger.setLevel(config.get("log_level", "INFO"))
    
    # Формат сообщений
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Консольный вывод
    if config.get("log_to_console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Файловый вывод
    if config.get("log_to_file", False):
        file_handler = logging.FileHandler("application.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config() -> dict:
    """
    Загружает конфигурацию приложения из JSON-файла
    
    :return: Словарь с конфигурацией
    :raises Exception: При ошибке загрузки файла
    """
    config_path = "configs/config.json"
    try:
        with open(config_path, "r") as config_file:
            return json.load(config_file)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки конфига {config_path}: {str(e)}")

def main():
    """Основная функция приложения"""
    try:
        # Загрузка конфигурации
        config = load_config()
        
        # Настройка логгера
        logger = setup_logger(config)
        logger.info("Приложение запущено")
        logger.debug(f"Загружен конфиг: {config}")
        
        # Создание интерфейса через фабрику
        interface_type = config.get("interface", "cli")
        logger.info(f"Создаем интерфейс типа: {interface_type}")
        
        interface = InterfaceFactory.create_interface(interface_type)
        
        # Запуск интерфейса
        logger.info("Запускаем интерфейс...")
        result = interface.run()
        
        # Вывод результата
        print("\n" + "=" * 50)
        print(f"РЕЗУЛЬТАТ: {result}")
        print("=" * 50)
        
        logger.info("Работа интерфейса завершена")
        
    except Exception as e:
        # Обработка критических ошибок
        error_msg = f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}"
        print(error_msg)
        if 'logger' in locals():
            logger.exception(error_msg)
        else:
            # Если логгер не успел инициализироваться
            with open("crash.log", "a") as f:
                f.write(error_msg + "\n")
    finally:
        if 'logger' in locals():
            logger.info("Приложение завершено")

if __name__ == "__main__":
    main()