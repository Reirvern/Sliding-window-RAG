# core/factories/synthesis_factory.py
import logging
from typing import Any
from core.domain.models import SynthesisConfig, InferenceConfig
from core.synthesis.base_synthesis import BaseSynthesis
from core.utils.localization.translator import Translator

# Импортируем конкретные реализации синтеза
from core.synthesis.simple_synthesis import SimpleSynthesis
# from core.synthesis.summarize_synthesis import SummarizeSynthesis # Пока не трогаем, но держим в уме

class SynthesisFactory:
    """
    Фабрика для создания различных стратегий синтеза.
    """
    _synthesis_map = {
        1: SimpleSynthesis,
        # 2: SummarizeSynthesis, # Для будущих стратегий
    }

    @staticmethod
    def get_synthesis_strategy(strategy_type: int,
                               config: SynthesisConfig,
                               inference_config: InferenceConfig,
                               logger: logging.Logger,
                               translator: Translator) -> BaseSynthesis:
        """
        Возвращает экземпляр конкретной стратегии синтеза.

        Args:
            strategy_type: Тип стратегии синтеза (например, 1 для SimpleSynthesis).
            config: Объект SynthesisConfig.
            inference_config: Объект InferenceConfig для модели синтеза.
            logger: Логгер.
            translator: Переводчик.

        Returns:
            Экземпляр BaseSynthesis.

        Raises:
            ValueError: Если указан неизвестный тип стратегии.
        """
        strategy_class = SynthesisFactory._synthesis_map.get(strategy_type)
        if not strategy_class:
            logger.error(f"Неизвестный тип стратегии синтеза: {strategy_type}")
            raise ValueError(f"Неизвестный тип стратегии синтеза: {strategy_type}")
        
        logger.info(f"Создаю стратегию синтеза: {strategy_class.__name__}")
        return strategy_class(config, inference_config, logger, translator)

