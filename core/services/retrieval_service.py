# core/services/retrieval_service.py
import logging
from typing import List
from core.domain.models import RAGQuery, RetrievalConfig, Chunk
from core.utils.observer import Observable

class RetrievalService(Observable):
    """
    Сервис для поиска релевантных чанков.
    """
    def __init__(self, config: RetrievalConfig, logger: logging.Logger, translator):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator # Нужен для промптов
        # self.retriever_factory = RetrieverFactory()
        # self.inference_engine = LlamacppInference() # Или другой инференс

    def retrieve(self, rag_query: RAGQuery, chunks: List[Chunk]) -> List[Chunk]:
        """
        Ищет релевантные чанки на основе запроса.
        """
        self.logger.info(f"Начинаю ретривинг с конфигом: стратегия {self.config.strategy_type}, top_k={self.config.top_k}")
        
        # TODO: Реальная логика вызова ретривера через фабрику
        # retriever = self.retriever_factory.get_retriever(self.config.strategy_type)
        # relevant_chunks = retriever.find_relevant(rag_query.question, chunks, self.config, self.inference_engine)
        
        relevant_chunks: List[Chunk] = []
        
        total_chunks = len(chunks)
        self.logger.debug(f"Всего чанков для поиска: {total_chunks}")

        # Имитация процесса ретривинга
        for i, chunk in enumerate(chunks):
            # Имитация логики: если вопрос встречается в чанке (или ключевые слова)
            if rag_query.question.lower() in chunk.content.lower() or \
               any(keyword.lower() in chunk.content.lower() for keyword in self.config.keywords):
                relevant_chunks.append(chunk)
            
            self.notify_observers("progress", {"stage": "retrieval", "current": i + 1, "total": total_chunks})
            
            # Ограничиваем количество найденных релевантных чанков
            if len(relevant_chunks) >= self.config.top_k:
                break
        
        self.logger.info(f"Ретривинг завершен. Найдено {len(relevant_chunks)} релевантных чанков.")
        self.notify_observers("complete", {"stage": "retrieval", "relevant_chunks_count": len(relevant_chunks)})
        return relevant_chunks