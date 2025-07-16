# core/factories/engine_factory.py
import logging
from core.engine.rag_engine import RAGEngine
from core.domain.models import RAGConfig # Импортируем RAGConfig

def create_rag_engine(rag_config: RAGConfig, logger: logging.Logger, translator) -> RAGEngine:
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
    try:
        # ИЗМЕНЕНО: Имя параметра изменено с 'config' на 'rag_config'
        return RAGEngine(config=rag_config, logger=logger, translator=translator)
    except Exception as e:
        logger.critical(f"Ошибка при создании RAG Engine: {e}", exc_info=True)
        raise
