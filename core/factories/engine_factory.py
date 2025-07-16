# core/factories/engine_factory.py
import logging
from core.engine.rag_engine import RAGEngine # Импортируем RAGEngine
from core.domain.models import RAGConfig # Импортируем RAGConfig
from core.utils.localization.translator import Translator

def create_rag_engine(rag_config: RAGConfig, logger: logging.Logger, translator: Translator):
    """
    Фабрика для создания экземпляра RAG Engine.
    
    Args:
        rag_config: Объект конфигурации RAG.
        logger: Логгер.
        translator: Переводчик.
        
    Returns:
        Экземпляр RAGEngine.
    """
    logger.info("Создание RAG Engine...")
    return RAGEngine(config=rag_config, logger=logger, translator=translator)