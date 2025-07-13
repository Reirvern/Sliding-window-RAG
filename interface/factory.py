from .cli import CLIInterface
from .gui import GUIInterface
from .server import ServerInterface
import logging

def create_interface(interface_type: str, config: dict, rag_engine, logger, translator):
    """
    Фабрика для создания интерфейсов
    
    Args:
        interface_type: Тип интерфейса (cli, gui, server)
        config: Конфигурация приложения
        rag_engine: Экземпляр RAG Engine
        logger: Логгер
        translator: Переводчик
        
    Returns:
        Экземпляр интерфейса
    """
    try:
        if interface_type == "cli":
            return CLIInterface(config, rag_engine, logger, translator)
        elif interface_type == "gui":
            logger.info("Создание GUI интерфейса...")
            return GUIInterface(config, rag_engine, logger, translator)
        elif interface_type == "server":
            logger.info("Создание серверного интерфейса...")
            return ServerInterface(config, rag_engine, logger, translator)
        else:
            raise ValueError(f"Неизвестный тип интерфейса: {interface_type}")
    except Exception as e:
        logger.error(f"Ошибка при создании интерфейса: {str(e)}")
        raise