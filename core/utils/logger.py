import logging

def setup_logger(config):
    """Простой логгер без сложностей"""
    logger = logging.getLogger("SWRAG")
    logger.setLevel(config.get("log_level", "INFO"))
    
    # Формат сообщений
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Консольный вывод
    if config.get("log_to_console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger