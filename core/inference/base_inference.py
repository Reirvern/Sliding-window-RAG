# core/inference/base_inference.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from core.domain.models import InferenceConfig

class BaseInferenceEngine(ABC):
    """
    Абстрактный базовый класс для всех инференс-движков.
    Определяет общий интерфейс для загрузки моделей и генерации текста.
    """
    def __init__(self, config: InferenceConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._loaded_model: Any = None # Для хранения загруженной модели

    @abstractmethod
    def load_model(self) -> Any:
        """
        Загружает модель в память.
        Должен быть идемпотентным: если модель уже загружена, возвращает её.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Генерирует текст на основе заданного промпта.
        
        Args:
            prompt: Входной промпт для генерации.
            kwargs: Дополнительные параметры генерации, переопределяющие конфиг.
            
        Returns:
            Сгенерированный текст.
        """
        pass

    @abstractmethod
    def unload_model(self):
        """
        Выгружает модель из памяти.
        """
        pass

    def _apply_generation_params(self, passed_kwargs: Dict[str, Any]) -> Dict[str, Any]: # ИЗМЕНЕНО: kwargs теперь passed_kwargs
        """
        Применяет параметры генерации из конфига, если они не переопределены.
        """
        final_params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "stop": self.config.stop_sequences,
        }
        final_params.update(passed_kwargs) # ИЗМЕНЕНО: используем passed_kwargs
        return final_params

