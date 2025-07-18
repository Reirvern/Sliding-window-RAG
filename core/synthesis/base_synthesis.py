# core/synthesis/base_synthesis.py
from abc import ABC, abstractmethod
import logging
from typing import List, Any
from core.domain.models import RAGQuery, Chunk, SynthesisConfig, InferenceConfig, SynthesisResult
from core.utils.observer import Observable
from core.utils.localization.translator import Translator

class BaseSynthesis(Observable, ABC):
    """
    Абстрактный базовый класс для всех стратегий синтеза.
    Определяет общий интерфейс для генерации финального ответа.
    """
    def __init__(self,
                 config: SynthesisConfig,
                 inference_config: InferenceConfig, # Конфиг для инференса модели синтеза
                 logger: logging.Logger,
                 translator: Translator):
        super().__init__()
        self.config = config
        self.inference_config = inference_config
        self.logger = logger
        self.translator = translator

    @abstractmethod
    def synthesize(self,
                   rag_query: RAGQuery,
                   relevant_chunks: List[Chunk],
                   inference_engine: Any) -> SynthesisResult:
        """
        Синтезирует финальный ответ на основе релевантных чанков и запроса.
        
        Args:
            rag_query: DTO пользовательского запроса.
            relevant_chunks: Список релевантных чанков.
            inference_engine: Экземпляр инференс-движка для генерации текста.
            
        Returns:
            Объект SynthesisResult, содержащий ответ и цитаты.
        """
        pass

