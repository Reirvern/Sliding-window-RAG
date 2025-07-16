# core/engine/rag_engine.py
import logging
from pathlib import Path
from typing import List, Dict, Any
from core.domain.models import RAGQuery, RAGConfig, Chunk # Импортируем все DTO
from core.services.chunking_service import ChunkingService
from core.services.retrieval_service import RetrievalService
from core.services.synthesis_service import SynthesisService
from core.utils.localization.translator import Translator
from core.utils.observer import Observable, Observer

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

        # Инициализация сервисов
        # Передаем им соответствующие части общего RAGConfig и логгер/транслятор
        self.chunking_service = ChunkingService(
            config=self.config.chunking,
            logger=self.logger
        )
        self.retrieval_service = RetrievalService(
            config=self.config.retrieval,
            logger=self.logger,
            translator=self.translator
        )
        self.synthesis_service = SynthesisService(
            config=self.config.synthesis,
            logger=self.logger,
            translator=self.translator
        )

        # Регистрация RAGEngine как Observer для своих сервисов
        # Это позволит RAGEngine получать прогресс от сервисов и передавать его дальше
        self.chunking_service.add_observer(self)
        self.retrieval_service.add_observer(self)
        self.synthesis_service.add_observer(self)


    def run(self, rag_query: RAGQuery) -> str:
        """
        Запускает полный цикл RAG для заданного запроса.
        :param rag_query: DTO, содержащий вопрос, путь к файлам и выходную папку.
        :return: Сгенерированный ответ.
        """
        self.logger.info(f"Начинаю RAG процесс для запроса: '{rag_query.question}'")
        self.notify_observers("status", {"message": self.translator.translate("rag_process_started")})

        try:
            # 1. Чанкинг
            self.logger.info("Шаг 1: Чанкинг документов...")
            self.notify_observers("status", {"message": self.translator.translate("chunking_in_progress")})
            chunks = self.chunking_service.process(rag_query)
            # TODO: Сохранить чанки в rag_query.output_dir / "chunks"

            if not chunks:
                self.logger.warning("Нет чанков для обработки.")
                self.notify_observers("complete", {"stage": "rag_process", "answer": self.translator.translate("no_chunks_to_process")})
                return self.translator.translate("no_chunks_to_process")

            # 2. Ретривинг
            self.logger.info("Шаг 2: Поиск релевантных чанков...")
            self.notify_observers("status", {"message": self.translator.translate("retrieval_in_progress")})
            relevant_chunks = self.retrieval_service.retrieve(rag_query, chunks)
            # TODO: Сохранить релевантные чанки в rag_query.output_dir / "relevant_chunks"

            if not relevant_chunks:
                self.logger.warning("Нет релевантных чанков для синтеза.")
                self.notify_observers("complete", {"stage": "rag_process", "answer": self.translator.translate("no_relevant_chunks")})
                return self.translator.translate("no_relevant_chunks")

            # 3. Синтез ответа
            self.logger.info("Шаг 3: Синтез финального ответа...")
            self.notify_observers("status", {"message": self.translator.translate("synthesis_in_progress")})
            final_answer = self.synthesis_service.synthesize_answer(rag_query, relevant_chunks)
            # TODO: Сохранить финальный ответ в rag_query.output_dir / "answer"

            self.logger.info("RAG процесс завершен успешно.")
            self.notify_observers("complete", {"stage": "rag_process", "answer": final_answer})
            return final_answer

        except Exception as e:
            self.logger.error(f"Ошибка в процессе RAG: {e}", exc_info=True)
            self.notify_observers("error", {"stage": "rag_process", "error": str(e)})
            return self.translator.translate("rag_process_error").format(error=str(e))

    def update(self, message_type: str, data: Any):
        """
        Реализация метода Observer для приема уведомлений от сервисов.
        Перенаправляет уведомления своим собственным наблюдателям (например, CLI).
        """
        self.notify_observers(message_type, data)
        # Дополнительно можно логировать или обрабатывать сообщения здесь
        self.logger.debug(f"RAGEngine получил уведомление: Type={message_type}, Data={data}")