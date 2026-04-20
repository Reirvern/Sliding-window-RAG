# interface/factory.py
import logging

def create_interface(interface_type: str, config: dict, rag_engine, logger, translator):
    """Фабрика для создания интерфейсов с ленивым импортом (Lazy Import)"""
    try:
        if interface_type == "cli":
            from .cli import CLIInterface
            return CLIInterface(config, rag_engine, logger, translator)
            
        elif interface_type == "gui":
            from .gui import GUIInterface
            logger.info("Создание GUI интерфейса...")
            return GUIInterface(config, rag_engine, logger, translator)
            
        elif interface_type == "webui":
            logger.info("WebUI запускается через собственный скрипт (Gradio).")
            class DummyInterface:
                def run(self): pass
            return DummyInterface()
            
        elif interface_type == "server":
            from .server import ServerInterface
            logger.info("Создание серверного интерфейса...")
            return ServerInterface(config, rag_engine, logger, translator)
            
        else:
            raise ValueError(f"Неизвестный тип интерфейса: {interface_type}")
    except Exception as e:
        logger.error(f"Ошибка при создании интерфейса: {str(e)}")
        raise