# core/factories/retriever_factory.py
import logging
from typing import Dict, Type

from core.retrieval.base_retriever import BaseRetriever
from core.retrieval.window_retriever import WindowRetriever # Импортируем новую стратегию
from core.retrieval.keyword_retriever import KeywordRetriever # Импортируем нашу стратегию
from core.domain.models import RetrievalConfig, InferenceConfig
from core.utils.localization.translator import Translator

class RetrieverFactory:
    """
    Фабрика для создания экземпляров различных стратегий ретривинга.
    """
    _retriever_map: Dict[int, Type[BaseRetriever]] = {
        1: WindowRetriever,  # Стратегия 1: Анализ каждого чанка с LLM
        2: KeywordRetriever, # Стратегия 2: Поиск по ключевым словам + LLM
        # Добавьте другие стратегии ретривинга здесь
    }

    @classmethod
    def get_retriever(cls, 
                      strategy_type: int, 
                      config: RetrievalConfig, 
                      inference_config: InferenceConfig, 
                      logger: logging.Logger,
                      translator: Translator) -> BaseRetriever:
        """
        Возвращает экземпляр ретривера на основе указанного типа стратегии.
        
        Args:
            strategy_type: Тип стратегии ретривинга (числовой идентификатор).
            config: Объект RetrievalConfig.
            inference_config: Объект InferenceConfig для ретривера.
            logger: Логгер.
            translator: Переводчик.
            
        Returns:
            Экземпляр BaseRetriever.
            
        Raises:
            ValueError: Если тип стратегии неизвестен.
        """
        retriever_class = cls._retriever_map.get(strategy_type)
        if not retriever_class:
            logger.error(f"Неизвестный тип стратегии ретривинга: {strategy_type}")
            raise ValueError(f"Неизвестный тип стратегии ретривинга: {strategy_type}")
        
        logger.info(f"Создание ретривера типа: {retriever_class.__name__}")
        return retriever_class(config=config, 
                               inference_config=inference_config, 
                               logger=logger,
                               translator=translator)

