# core/services/synthesis_service.py
import logging
from typing import List, Any, Dict # Добавляем Dict
from core.domain.models import RAGQuery, Chunk, SynthesisConfig, InferenceConfig, SynthesisResult # Импортируем SynthesisResult
from core.utils.observer import Observable, Observer
from core.utils.localization.translator import Translator
from core.factories.synthesis_factory import SynthesisFactory # Импортируем фабрику синтеза

class SynthesisService(Observable, Observer):
    """
    Сервис для синтеза финального ответа на основе релевантных чанков.
    """
    def __init__(self, 
                 config: SynthesisConfig, 
                 logger: logging.Logger, 
                 translator: Translator,
                 inference_engine):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator
        self.inference_engine = inference_engine # Инференс-движок для синтеза

        # Инициализируем стратегию синтеза через фабрику
        self.synthesis_strategy = SynthesisFactory.get_synthesis_strategy(
            strategy_type=self.config.strategy_type,
            config=self.config,
            inference_config=self.inference_engine.config, # Передаем конфиг инференса синтеза
            logger=self.logger,
            translator=self.translator
        )
        # Регистрируем SynthesisService как наблюдателя для стратегии синтеза
        self.synthesis_strategy.add_observer(self)

    def synthesize_answer(self, rag_query: RAGQuery, relevant_chunks: List[Chunk]) -> SynthesisResult: # Возвращаем SynthesisResult
        """
        Синтезирует финальный ответ, используя релевантные чанки и LLM.
        """
        self.logger.info("Начинаю синтез ответа.")
        self.notify_observers("status", {"message": self.translator.translate("synthesis_in_progress")})

        try:
            # Вызываем метод synthesize у конкретной стратегии синтеза
            final_result = self.synthesis_strategy.synthesize(rag_query, relevant_chunks, self.inference_engine)
            self.logger.info("Синтез ответа завершен.")
            self.notify_observers("complete", {"stage": "synthesis", "answer_length": len(final_result.answer), "citations_count": len(final_result.citations)})
            return final_result
        except Exception as e:
            self.logger.error(f"Ошибка при синтезе финального ответа: {e}", exc_info=True)
            self.notify_observers("error", {"stage": "synthesis", "error": str(e)})
            # Возвращаем SynthesisResult с ошибкой
            return SynthesisResult(answer=self.translator.translate("synthesis_error").format(error=str(e)), citations=[])

    def update(self, message_type: str, data: Any):
        """
        Реализация метода Observer для приема уведомлений от стратегии синтеза.
        Перенаправляет уведомления своим собственным наблюдателям (например, RAGEngine).
        """
        self.notify_observers(message_type, data)
        if message_type != "progress":
            self.logger.debug(f"SynthesisService получил уведомление: Type={message_type}, Data={data}")

