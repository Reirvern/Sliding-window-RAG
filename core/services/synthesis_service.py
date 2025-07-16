# core/services/synthesis_service.py
import logging
from typing import List
from core.domain.models import RAGQuery, SynthesisConfig, Chunk
from core.utils.observer import Observable
from core.inference.base_inference import BaseInferenceEngine # Импортируем базовый класс движка

class SynthesisService(Observable):
    """
    Сервис для генерации финального ответа на основе релевантных чанков.
    """
    def __init__(self, config: SynthesisConfig, logger: logging.Logger, translator, inference_engine: BaseInferenceEngine):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator
        self.inference_engine = inference_engine # Принимаем инференс-движок

    def synthesize_answer(self, rag_query: RAGQuery, relevant_chunks: List[Chunk]) -> str:
        """
        Генерирует финальный ответ на основе релевантных чанков.
        """
        self.logger.info(f"Начинаю синтез ответа.")
        
        if not relevant_chunks:
            self.logger.warning("Нет релевантных чанков для синтеза ответа.")
            self.notify_observers("complete", {"stage": "synthesis", "answer": self.translator.translate("no_relevant_chunks")})
            return self.translator.translate("no_relevant_chunks")
        
        # Собираем контекст из релевантных чанков
        context = "\n---\n".join([chunk.content for chunk in relevant_chunks])
        
        # Формируем промпт для LLM, используя шаблон из SynthesisConfig
        prompt = self.config.prompt_template.format(
            question=rag_query.question,
            context=context
        )
        
        # Загружаем модель для синтеза (если еще не загружена)
        self.inference_engine.load_model()

        self.logger.debug(f"Промпт для генерации (начало): {prompt[:200]}...")
        
        # Используем инференс-движок для генерации ответа
        # Здесь можно передать параметры генерации, если они отличаются от конфига движка
        final_answer_raw = self.inference_engine.generate(prompt)
                       
        self.logger.info("Синтез ответа завершен.")
        # Для демонстрации прогресса, можно было бы сделать это в цикле, если движок поддерживает потоковую генерацию.
        self.notify_observers("complete", {"stage": "synthesis", "answer": final_answer_raw})
        return self.translator.translate("final_answer_prefix") + " " + final_answer_raw + " " + \
               self.translator.translate("summary_from_chunks").format(chunks=len(relevant_chunks)) + \
               f"\n\nВопрос: {rag_query.question}"