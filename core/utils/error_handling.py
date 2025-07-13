# core/utils/error_handling.py
import logging

def log_unhandled_exception(logger, exception: Exception):
    logger.error(f"Необработанное исключение: {str(exception)}")
    logger.exception("Детали исключения:")