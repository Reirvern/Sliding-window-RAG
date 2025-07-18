# core/engine/rag_engine.py
import logging
from pathlib import Path
from typing import List, Dict, Any
from core.domain.models import RAGQuery, RAGConfig, Chunk, SynthesisResult # Импортируем SynthesisResult
from core.services.chunking_service import ChunkingService
from core.services.retrieval_service import RetrievalService
from core.services.synthesis_service import SynthesisService
from core.utils.localization.translator import Translator
from core.utils.observer import Observable, Observer
from core.factories.inference_factory import InferenceFactory # Импортируем фабрику инференса

class RAGEngine(Observable):
    """
    Главный координатор процесса Retrieval-Augmented Generation (RAG).
    Оркестрирует работу ChunkingService, RetrievalService и SynthesisService.
    """
    def __init__(self, config: RAGConfig, logger: logging.Logger, translator: Translator):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator

        # Инициализация инференс-движков (они пока не загружены)
        self.retrieval_inference_engine = InferenceFactory.get_engine(
            config=self.config.retrieval_inference,
            logger=self.logger
        )
        self.synthesis_inference_engine = InferenceFactory.get_engine(
            config=self.config.synthesis_inference,
            logger=self.logger
        )
        # НОВОЕ: Инициализация инференс-движка для запасного ретривера, если он сконфигурирован
        self.retrieval_fallback_inference_engine = None
        if self.config.retrieval_fallback_inference:
            self.retrieval_fallback_inference_engine = InferenceFactory.get_engine(
                config=self.config.retrieval_fallback_inference,
                logger=self.logger
            )

        # Инициализация сервисов
        self.chunking_service = ChunkingService(
            config=self.config.chunking,
            logger=self.logger
        )
        self.retrieval_service = RetrievalService(
            config=self.config.retrieval,
            logger=self.logger,
            translator=self.translator,
            inference_engine=self.retrieval_inference_engine, # Передаем основной движок
            fallback_inference_engine=self.retrieval_fallback_inference_engine # НОВОЕ: Передаем запасной движок
        )
        self.synthesis_service = SynthesisService(
            config=self.config.synthesis,
            logger=self.logger,
            translator=self.translator,
            inference_engine=self.synthesis_inference_engine
        )

        # Регистрируем RAGEngine как наблюдателя для каждого сервиса
        self.chunking_service.add_observer(self)
        self.retrieval_service.add_observer(self)
        self.synthesis_service.add_observer(self)

    def run(self, rag_query: RAGQuery) -> SynthesisResult: # Возвращаем SynthesisResult
        """
        Запускает полный процесс RAG: чанкинг, ретривинг, синтез.
        """
        self.logger.info(f"Начинаю RAG процесс для запроса: '{rag_query.question}'")
        self.notify_observers("status", {"message": self.translator.translate("rag_process_started")})

        try:
            # Шаг 1: Чанкинг документов
            self.logger.info("Шаг 1: Чанкинг документов...")
            self.notify_observers("status", {"message": self.translator.translate("chunking_in_progress")})
            chunks = self.chunking_service.process(rag_query)
            if not chunks:
                self.logger.warning("Чанкинг не дал результатов. Пропускаю дальнейшие шаги.")
                self.notify_observers("complete", {"stage": "rag_process", "answer": self.translator.translate("no_chunks_to_process")})
                return SynthesisResult(answer=self.translator.translate("no_chunks_to_process"), citations=[])

            # Шаг 2: Поиск релевантных чанков
            self.logger.info("Шаг 2: Поиск релевантных чанков...")
            self.notify_observers("status", {"message": self.translator.translate("retrieval_in_progress")})
            relevant_chunks = self.retrieval_service.retrieve(rag_query, chunks)
            if not relevant_chunks:
                self.logger.warning("Ретривинг не нашел релевантных чанков. Пропускаю синтез.")
                self.notify_observers("complete", {"stage": "rag_process", "answer": self.translator.translate("no_relevant_chunks")})
                return SynthesisResult(answer=self.translator.translate("no_relevant_chunks"), citations=[])

            # Шаг 3: Синтез финального ответа
            self.logger.info("Шаг 3: Синтез финального ответа...")
            self.notify_observers("status", {"message": self.translator.translate("synthesis_in_progress")})
            final_result = self.synthesis_service.synthesize_answer(rag_query, relevant_chunks) # Получаем SynthesisResult

            self.logger.info("RAG процесс завершен успешно.")
            self.notify_observers("complete", {"stage": "rag_process", "answer": final_result.answer}) # Передаем только строку ответа для общего уведомления
            return final_result

        except Exception as e:
            self.logger.error(f"Ошибка в процессе RAG: {e}", exc_info=True)
            self.notify_observers("error", {"stage": "rag_process", "error": str(e)})
            final_answer_error = self.translator.translate("rag_process_error").format(error=str(e))
            return SynthesisResult(answer=final_answer_error, citations=[]) # Возвращаем SynthesisResult с ошибкой
        finally:
            # Убедимся, что все модели выгружены
            if self.retrieval_inference_engine._loaded_model:
                self.retrieval_inference_engine.unload_model()
            if self.synthesis_inference_engine._loaded_model:
                self.synthesis_inference_engine.unload_model()
            # НОВОЕ: Выгружаем запасную модель, если она была загружена
            if self.retrieval_fallback_inference_engine and self.retrieval_fallback_inference_engine._loaded_model:
                self.retrieval_fallback_inference_engine.unload_model()


    def update(self, message_type: str, data: Any):
        """
        Реализация метода Observer для приема уведомлений от сервисов.
        Перенаправляет уведомления своим собственным наблюдателям (например, CLI).
        """
        self.notify_observers(message_type, data)
        # ИЗМЕНЕНО: Логируем DEBUG сообщения только для не-прогресс обновлений
        if message_type != "progress":
            self.logger.debug(f"RAGEngine получил уведомление: Type={message_type}, Data={data}")

