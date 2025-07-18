# core/retrieval/window_retriever.py
import logging
from typing import List, Pattern
from pathlib import Path
import json
import re

from core.retrieval.base_retriever import BaseRetriever
from core.domain.models import RAGQuery, Chunk, RetrievalConfig, InferenceConfig
from core.utils.localization.translator import Translator

class WindowRetriever(BaseRetriever):
    """
    Стратегия ретривинга, которая анализирует каждый чанк с помощью LLM
    для определения его релевантности.
    """
    def __init__(self, 
                 config: RetrievalConfig, 
                 inference_config: InferenceConfig, 
                 logger: logging.Logger,
                 translator: Translator):
        super().__init__(config, inference_config, logger)
        self.translator = translator
        self.yes_pattern: Pattern = re.compile(r'\b(да|yes)\b', re.IGNORECASE)
        self.no_pattern: Pattern = re.compile(r'\b(нет|no)\b', re.IGNORECASE)

    def retrieve(self, 
                 rag_query: RAGQuery, 
                 chunks: List[Chunk],
                 inference_engine) -> List[Chunk]:
        """
        Ищет релевантные чанки, анализируя каждый чанк с помощью LLM.
        """
        self.logger.info(f"Начинаю ретривинг с WindowRetriever. Всего чанков: {len(chunks)}")
        self.notify_observers("status", {"message": self.translator.translate("retrieval_strategy_start").format(strategy_type=self.config.strategy_type)})

        relevant_chunks: List[Chunk] = []
        retrieved_count = 0
        total_chunks = len(chunks)

        relevant_chunks_output_dir = rag_query.output_dir / "relevant_chunks"
        relevant_chunks_output_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks):
            self.notify_observers("progress", {
                "stage": "retrieval",
                "current": i + 1,
                "total": total_chunks
            })

            self.logger.debug(f"Отправляю чанк {chunk.chunk_id} в LLM для оценки релевантности.")
            try:
                # Формируем список сообщений для create_chat_completion
                messages = [
                    {"role": "user", "content": f"Основываясь только на следующем фрагменте текста, ответь на вопрос '{rag_query.question}' одним словом: 'да' или 'нет'.\n\nФрагмент: {chunk.content}\n\nОтвет:"}
                ]

                llm_response = inference_engine.generate(
                    messages=messages,
                    temperature=self.inference_config.temperature,
                    max_new_tokens=self.inference_config.max_new_tokens,
                    top_p=self.inference_config.top_p,
                    top_k=self.inference_config.top_k,
                    repeat_penalty=self.inference_config.repeat_penalty,
                    stop=self.inference_config.stop_sequences
                )
                
                processed_llm_response = self._extract_yes_no(llm_response)
                self.logger.debug(f"Ответ LLM для чанка {chunk.chunk_id} (сырой): '{llm_response.strip()}'")
                self.logger.debug(f"Ответ LLM для чанка {chunk.chunk_id} (обработанный): '{processed_llm_response}'")

                if processed_llm_response == "да":
                    relevant_chunks.append(chunk)
                    retrieved_count += 1
                    self.logger.info(f"Чанк {chunk.chunk_id} помечен как релевантный. Всего релевантных: {retrieved_count}")

                    relevant_chunk_data = {
                        "chunk_id": chunk.chunk_id,
                        "file_name": chunk.file_path.name,
                        "original_file_path": str(chunk.file_path),
                        "content": chunk.content,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "length": len(chunk.content),
                        "metadata": chunk.metadata,
                        "llm_relevance_assessment": llm_response.strip(),
                        "processed_llm_assessment": processed_llm_response,
                        "query": rag_query.question
                    }
                    relevant_chunk_file_path = relevant_chunks_output_dir / f"relevant_chunk_{chunk.chunk_id}.json"
                    with open(relevant_chunk_file_path, 'w', encoding='utf-8') as f:
                        json.dump(relevant_chunk_data, f, ensure_ascii=False, indent=4)
                    self.logger.debug(f"Релевантный чанк {chunk.chunk_id} сохранен в {relevant_chunk_file_path.name}")

                    if self.config.top_k > 0 and retrieved_count >= self.config.top_k:
                        self.logger.info(f"Достигнуто top_k ({self.config.top_k}) релевантных чанков. Завершаю поиск.")
                        break
                elif processed_llm_response == "нет":
                    self.logger.debug(f"Чанк {chunk.chunk_id} помечен как нерелевантный.")
                else:
                    self.logger.warning(f"Неоднозначный или некорректный ответ LLM для чанка {chunk.chunk_id}: '{llm_response.strip()}'. Обработанный: '{processed_llm_response}'. Чанк пропущен.")

            except Exception as e:
                self.logger.error(f"Ошибка при оценке релевантности чанка {chunk.chunk_id} с помощью LLM: {e}", exc_info=True)
                self.notify_observers("error", {"stage": "retrieval", "error": f"Ошибка LLM для чанка {chunk.chunk_id}: {str(e)}"})
                continue

        self.logger.info(f"Ретривинг завершен. Найдено {len(relevant_chunks)} релевантных чанков.")
        self.notify_observers("complete", {"stage": "retrieval", "relevant_chunks_count": len(relevant_chunks)})
        return relevant_chunks

    def _extract_yes_no(self, text: str) -> str:
        text_lower = text.strip().lower()
        if self.yes_pattern.search(text_lower):
            return "да"
        if self.no_pattern.search(text_lower):
            return "нет"
        return ""
