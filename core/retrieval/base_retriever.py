# core/retrieval/base_retriever.py
import logging
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from core.domain.models import RAGQuery, Chunk, RetrievalConfig, InferenceConfig
from core.utils.observer import Observable # BaseRetriever будет наблюдаемым

class BaseRetriever(ABC, Observable):
    """
    Абстрактный базовый класс для всех стратегий ретривинга.
    Все конкретные ретриверы должны наследоваться от него.
    """
    def __init__(self, 
                 config: RetrievalConfig, 
                 inference_config: InferenceConfig, # Конфиг для инференса ретривера
                 logger: logging.Logger):
        super().__init__()
        self.config = config
        self.inference_config = inference_config
        self.logger = logger

    @abstractmethod
    def retrieve(self, 
                 rag_query: RAGQuery, 
                 chunks: List[Chunk],
                 inference_engine) -> List[Chunk]:
        """
        Абстрактный метод для поиска релевантных чанков.
        
        Args:
            rag_query: Объект RAGQuery, содержащий вопрос и пути.
            chunks: Список всех доступных чанков.
            inference_engine: Экземпляр инференс-движка для взаимодействия с моделью.
            
        Returns:
            Список релевантных чанков.
        """
        pass

