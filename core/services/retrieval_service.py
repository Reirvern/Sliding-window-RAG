# core/services/retrieval_service.py
import logging
from typing import List
from core.domain.models import RAGQuery, Chunk, RetrievalConfig, InferenceConfig # Импортируем InferenceConfig
from core.utils.observer import Observable, Observer
from typing import Any
from core.factories.retriever_factory import RetrieverFactory # Импортируем фабрику ретриверов
from core.utils.localization.translator import Translator

class RetrievalService(Observable, Observer): # RetrievalService также может быть Observer для InferenceEngine
    """
    Сервис для поиска наиболее релевантных чанков на основе запроса.
    """
    def __init__(self, 
                 config: RetrievalConfig, 
                 logger: logging.Logger, 
                 translator: Translator,
                 inference_engine): # Принимаем инференс-движок
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator
        self.inference_engine = inference_engine # Сохраняем инференс-движок

        # Фабрика ретриверов для получения нужной стратегии
        # RetrieverFactory.get_retriever теперь требует inference_config
        # Передаем inference_config из RAGEngine через RetrievalService
        self.retriever = RetrieverFactory.get_retriever(
            strategy_type=self.config.strategy_type,
            config=self.config,
            inference_config=self.inference_engine.config, # Передаем конфиг инференса ретривера
            logger=self.logger,
            translator=self.translator
        )
        # Регистрируем RetrievalService как наблюдателя для ретривера
        self.retriever.add_observer(self) 

    def retrieve(self, rag_query: RAGQuery, chunks: List[Chunk]) -> List[Chunk]:
        """
        Выполняет поиск релевантных чанков, используя выбранную стратегию.
        """
        self.logger.info("Начинаю процесс ретривинга...")
        self.notify_observers("status", {"message": self.translator.translate("retrieval_in_progress")})
        
        # Вызываем метод retrieve у конкретной стратегии ретривера
        relevant_chunks = self.retriever.retrieve(rag_query, chunks, self.inference_engine)
        
        self.logger.info(f"Ретривинг завершен. Найдено {len(relevant_chunks)} релевантных чанков.")
        self.notify_observers("complete", {"stage": "retrieval", "relevant_chunks_count": len(relevant_chunks)})
        return relevant_chunks

    def update(self, message_type: str, data: Any):
        """
        Реализация метода Observer для приема уведомлений от ретривера.
        Перенаправляет уведомления своим собственным наблюдателям (например, RAGEngine).
        """
        self.notify_observers(message_type, data)
        # Логируем DEBUG сообщения только для не-прогресс обновлений
        if message_type != "progress":
            self.logger.debug(f"RetrievalService получил уведомление: Type={message_type}, Data={data}")

