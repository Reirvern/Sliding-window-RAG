# core/services/synthesis_service.py
import logging
from typing import List
from core.domain.models import RAGQuery, SynthesisConfig, Chunk
from core.utils.observer import Observable

class SynthesisService(Observable):
    """
    Сервис для генерации финального ответа на основе релевантных чанков.
    """
    def __init__(self, config: SynthesisConfig, logger: logging.Logger, translator):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator # Нужен для промптов
        # self.inference_engine = LlamacppInference() # Или другой инференс

    def synthesize_answer(self, rag_query: RAGQuery, relevant_chunks: List[Chunk]) -> str:
        """
        Генерирует финальный ответ на основе релевантных чанков.
        """
        self.logger.info(f"Начинаю синтез ответа с конфигом: temp={self.config.temperature}, max_tokens={self.config.max_new_tokens}")
        
        if not relevant_chunks:
            self.logger.warning("Нет релевантных чанков для синтеза ответа.")
            self.notify_observers("complete", {"stage": "synthesis", "answer": self.translator.translate("no_relevant_chunks")})
            return self.translator.translate("no_relevant_chunks")
        
        # Собираем контекст из релевантных чанков
        context = "\n---\n".join([chunk.content for chunk in relevant_chunks])
        
        # Формируем промпт для LLM
        prompt = self.translator.translate("synthesis_prompt_template").format(
            question=rag_query.question,
            context=context
        )
        
        # TODO: Реальная логика вызова LLM
        # answer = self.inference_engine.generate(prompt, self.config)
        
        # Имитация генерации ответа
        self.logger.debug(f"Промпт для генерации: {prompt[:200]}...")
        total_steps = 10 # Имитация шагов генерации
        generated_answer = ""
        for i in range(total_steps):
            generated_answer += f"Часть {i+1} ответа... "
            self.notify_observers("progress", {"stage": "synthesis", "current": i + 1, "total": total_steps})
            
        final_answer = self.translator.translate("final_answer_prefix") + " " + generated_answer + " " + \
                       self.translator.translate("summary_from_chunks").format(chunks=len(relevant_chunks)) + \
                       f"\n\nВопрос: {rag_query.question}"
                       
        self.logger.info("Синтез ответа завершен.")
        self.notify_observers("complete", {"stage": "synthesis", "answer": final_answer})
        return final_answer