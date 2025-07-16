# core/factories/inference_factory.py
import logging
from typing import Dict, Type

from core.inference.base_inference import BaseInferenceEngine
from core.inference.llamacpp_inference import LlamacppInferenceEngine
from core.domain.models import InferenceConfig

class InferenceFactory:
    """
    Фабрика для создания экземпляров инференс-движков.
    """
    _engine_map: Dict[str, Type[BaseInferenceEngine]] = {
        "llamacpp": LlamacppInferenceEngine,
        # TODO: Добавьте сюда другие движки, например, "vllm_stub": VllmInferenceEngineStub,
    }

    @classmethod
    def get_engine(cls, config: InferenceConfig, logger: logging.Logger) -> BaseInferenceEngine:
        """
        Возвращает экземпляр инференс-движка на основе конфигурации.
        
        Args:
            config: Конфигурация инференс-движка.
            logger: Логгер.
            
        Returns:
            Экземпляр BaseInferenceEngine или его подкласса.
            
        Raises:
            ValueError: Если тип движка не поддерживается.
        """
        engine_class = cls._engine_map.get(config.engine_type)

        if not engine_class:
            logger.error(f"Неизвестный тип инференс-движка: {config.engine_type}")
            raise ValueError(f"Неизвестный тип инференс-движка: {config.engine_type}")
        
        return engine_class(config, logger)