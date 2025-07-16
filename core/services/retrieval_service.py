# core/services/retrieval_service.py
import logging
from typing import List
from core.domain.models import RAGQuery, RetrievalConfig, Chunk
from core.utils.observer import Observable
from core.inference.base_inference import BaseInferenceEngine # Импортируем базовый класс движка

class RetrievalService(Observable):
    """
    Сервис для поиска релевантных чанков.
    """
    def __init__(self, config: RetrievalConfig, logger: logging.Logger, translator, inference_engine: BaseInferenceEngine):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator
        self.inference_engine = inference_engine # Принимаем инференс-движок

    def retrieve(self, rag_query: RAGQuery, chunks: List[Chunk]) -> List[Chunk]:
        """
        Ищет релевантные чанки на основе запроса.
        """
        self.logger.info(f"Начинаю ретривинг с конфигом: стратегия {self.config.strategy_type}, top_k={self.config.top_k}")
        
        relevant_chunks: List[Chunk] = []
        
        total_chunks = len(chunks)
        if total_chunks == 0:
            self.logger.warning("Нет чанков для ретривинга.")
            self.notify_observers("complete", {"stage": "retrieval", "relevant_chunks_count": 0})
            return []

        self.logger.debug(f"Всего чанков для поиска: {total_chunks}")

        # Имитация процесса ретривинга с использованием LLM для оценки релевантности
        # В реальной реализации здесь будет более сложная логика, возможно, с векторным поиском
        # и последующей переранжировкой LLM.
        
        # Загружаем модель для ретривинга (если еще не загружена)
        self.inference_engine.load_model()

        for i, chunk in enumerate(chunks):
            # Пример промпта для LLM для оценки релевантности
            relevance_prompt = self.translator.translate("retrieval_relevance_prompt").format(
                question=rag_query.question,
                chunk_content=chunk.content
            )
            
            # Используем инференс-движок для получения ответа (Да/Нет или оценка)
            # Для простоты, имитируем ответ LLM
            # llm_response = self.inference_engine.generate(relevance_prompt, max_new_tokens=10, temperature=0.1)
            
            # Имитация: если вопрос или ключевые слова есть в чанке, считаем релевантным
            is_relevant = False
            if rag_query.question.lower() in chunk.content.lower():
                is_relevant = True
            elif self.config.keywords:
                if any(keyword.lower() in chunk.content.lower() for keyword in self.config.keywords):
                    is_relevant = True

            if is_relevant:
                relevant_chunks.append(chunk)
            
            self.notify_observers("progress", {"stage": "retrieval", "current": i + 1, "total": total_chunks})
            
            # Ограничиваем количество найденных релевантных чанков
            if len(relevant_chunks) >= self.config.top_k:
                self.logger.info(f"Достигнуто максимальное количество релевантных чанков ({self.config.top_k}). Завершаю поиск.")
                break
        
        self.logger.info(f"Ретривинг завершен. Найдено {len(relevant_chunks)} релевантных чанков.")
        self.notify_observers("complete", {"stage": "retrieval", "relevant_chunks_count": len(relevant_chunks)})
        return relevant_chunks