# core/factories/retriever_factory.py
import logging
from typing import Any, Optional

from core.retrieval.base_retriever import BaseRetriever
from core.domain.models import RetrievalConfig, InferenceConfig
from core.utils.localization.translator import Translator

# Импортируем конкретные реализации ретриверов
from core.retrieval.window_retriever import WindowRetriever
from core.retrieval.keyword_retriever import KeywordRetriever
from core.retrieval.best_window_retriever import BestWindowRetriever # НОВОЕ: Импортируем BestWindowRetriever

class RetrieverFactory:
    """
    Фабрика для создания различных стратегий ретривинга.
    """
    # НОВОЕ: Добавляем BestWindowRetriever
    _retriever_map = {
        1: WindowRetriever, # Теперь 1 будет BestWindowRetriever
        2: KeywordRetriever,
        3: BestWindowRetriever,
        # Если захотим оставить старый WindowRetriever, можно добавить 3: WindowRetriever
    }

    @staticmethod
    def get_retriever(strategy_type: int,
                      config: RetrievalConfig,
                      inference_config: InferenceConfig, # Конфиг основной модели ретривера
                      logger: logging.Logger,
                      translator: Translator,
                      fallback_inference_config: Optional[InferenceConfig] = None, # НОВОЕ: Конфиг запасной модели
                      fallback_inference_engine: Optional[Any] = None # НОВОЕ: Экземпляр запасного инференс-движка
                      ) -> BaseRetriever:
        """
        Возвращает экземпляр конкретной стратегии ретривера.

        Args:
            strategy_type: Тип стратегии ретривера (например, 1 для BestWindowRetriever).
            config: Объект RetrievalConfig.
            inference_config: Объект InferenceConfig для основной модели ретривера.
            logger: Логгер.
            translator: Переводчик.
            fallback_inference_config: Объект InferenceConfig для запасной модели ретривера (опционально).
            fallback_inference_engine: Экземпляр инференс-движка для запасной модели ретривера (опционально).

        Returns:
            Экземпляр BaseRetriever.

        Raises:
            ValueError: Если указан неизвестный тип стратегии.
        """
        strategy_class = RetrieverFactory._retriever_map.get(strategy_type)
        if not strategy_class:
            logger.error(f"Неизвестный тип стратегии ретривинга: {strategy_type}")
            raise ValueError(f"Неизвестный тип стратегии ретривинга: {strategy_type}")
        
        logger.info(f"Создаю стратегию ретривинга: {strategy_class.__name__}")
        
        # НОВОЕ: Передаем параметры запасной модели, если они есть
        if strategy_class == BestWindowRetriever:
            return strategy_class(config, inference_config, logger, translator,
                                  fallback_inference_config=fallback_inference_config,
                                  fallback_inference_engine=fallback_inference_engine)
        else:
            return strategy_class(config, inference_config, logger, translator)

