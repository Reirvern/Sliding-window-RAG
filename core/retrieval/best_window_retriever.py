# core/retrieval/best_window_retriever.py
import logging
from typing import List, Pattern, Any, Optional, Dict
from pathlib import Path
import json
import re

from core.retrieval.base_retriever import BaseRetriever
from core.domain.models import RAGQuery, Chunk, RetrievalConfig, InferenceConfig
from core.utils.localization.translator import Translator

class BestWindowRetriever(BaseRetriever):
    """
    Продвинутая стратегия ретривинга, которая анализирует каждый чанк с помощью LLM
    для определения его релевантности, используя каскад моделей и строгие правила.
    """
    def __init__(self, 
                 config: RetrievalConfig, 
                 inference_config: InferenceConfig, # Конфиг основной модели ретривера
                 logger: logging.Logger,
                 translator: Translator,
                 fallback_inference_config: Optional[InferenceConfig] = None, # Конфиг запасной модели
                 fallback_inference_engine: Optional[Any] = None # Экземпляр запасного инференс-движка
                 ):
        super().__init__(config, inference_config, logger)
        self.translator = translator
        # Паттерны для поиска 'да' или 'нет' в ответе (как в исходном WindowRetriever)
        self.yes_pattern: Pattern = re.compile(r'\b(да|yes)\b', re.IGNORECASE)
        self.no_pattern: Pattern = re.compile(r'\b(нет|no)\b', re.IGNORECASE)

        self.fallback_inference_config = fallback_inference_config
        self.fallback_inference_engine = fallback_inference_engine
        self.logger.debug(f"BestWindowRetriever инициализирован. Наличие fallback модели: {self.fallback_inference_engine is not None}")

    def retrieve(self, 
                 rag_query: RAGQuery, 
                 chunks: List[Chunk],
                 inference_engine) -> List[Chunk]:
        """
        Ищет релевантные чанки, анализируя каждый чанк с помощью LLM.
        Использует каскад моделей и строгие правила для определения релевантности.
        """
        self.logger.info(f"Начинаю ретривинг с BestWindowRetriever. Всего чанков: {len(chunks)}")
        self.notify_observers("status", {"message": self.translator.translate("retrieval_in_progress")})

        relevant_chunks: List[Chunk] = []
        retrieved_count = 0
        total_chunks = len(chunks)

        relevant_chunks_output_dir = rag_query.output_dir / "relevant_chunks"
        relevant_chunks_output_dir.mkdir(parents=True, exist_ok=True)

        # Загружаем основную модель для ретривинга
        inference_engine.load_model()

        try: # Этот try-блок теперь охватывает весь цикл и выгрузку
            for i, chunk in enumerate(chunks):
                self.notify_observers("progress", {
                    "stage": "retrieval",
                    "current": i + 1,
                    "total": total_chunks,
                    "message": self.translator.translate("progress_retrieval_overall")
                })

                self.logger.debug(f"Отправляю чанк {chunk.chunk_id} в основную LLM для оценки релевантности.")
                
                # Попытка 1: Основная модель с обычными настройками и промптом
                # Формируем промпт для LLM (как в оригинальном WindowRetriever)
                prompt_attempt1 = self.config.retriever_prompt.format(
                    prompt=rag_query.question,
                    chunk_content=chunk.content
                )
                messages_attempt1 = [{"role": "user", "content": prompt_attempt1}]

                llm_response_attempt1 = inference_engine.generate(
                    messages=messages_attempt1,
                    temperature=self.inference_config.temperature,
                    max_new_tokens=self.inference_config.max_new_tokens,
                    top_p=self.inference_config.top_p,
                    top_k=self.inference_config.top_k,
                    repeat_penalty=self.inference_config.repeat_penalty,
                    stop=self.inference_config.stop_sequences # Используем стоп-последовательности из конфига основной модели
                )
                processed_llm_response = self._extract_yes_no(llm_response_attempt1)
                self.logger.debug(f"Ответ LLM (попытка 1) для чанка {chunk.chunk_id} (сырой): '{llm_response_attempt1.strip()}', обработанный: '{processed_llm_response}'")

                # Если ответ не "да" или "нет", и есть запасная модель, запускаем каскад
                if processed_llm_response not in ["да", "нет"] and self.fallback_inference_engine:
                    # Логируем, как в старом WindowRetriever, но затем пытаемся исправить
                    self.logger.warning(f"Неоднозначный или некорректный ответ LLM для чанка {chunk.chunk_id}: '{llm_response_attempt1.strip()}'. Обработанный: '{processed_llm_response}'. Пробую запасную модель.")
                    self.notify_observers("status", {"message": self.translator.translate("retrieval_fallback_attempt").format(chunk_id=chunk.chunk_id)})

                    # Загружаем запасную модель
                    self.fallback_inference_engine.load_model()

                    # Попытка 2: Запасная модель с более строгими настройками из конфига
                    prompt_attempt2 = self.config.retriever_fallback_prompt.format(
                        prompt=rag_query.question,
                        chunk_content=chunk.content
                    )
                    messages_attempt2 = [{"role": "user", "content": prompt_attempt2}]

                    llm_response_fallback = self.fallback_inference_engine.generate(
                        messages=messages_attempt2,
                        temperature=self.fallback_inference_config.temperature,
                        max_new_tokens=self.fallback_inference_config.max_new_tokens,
                        top_p=self.fallback_inference_config.top_p,
                        top_k=self.fallback_inference_config.top_k,
                        repeat_penalty=self.fallback_inference_config.repeat_penalty,
                        stop=self.fallback_inference_config.stop_sequences # Здесь используем стоп-последовательности
                    )
                    processed_llm_response_fallback = self._extract_yes_no(llm_response_fallback)
                    self.logger.debug(f"Ответ LLM (попытка 2, fallback) для чанка {chunk.chunk_id} (сырой): '{llm_response_fallback.strip()}', обработанный: '{processed_llm_response_fallback}'")

                    if processed_llm_response_fallback not in ["да", "нет"]:
                        self.logger.warning(f"Запасная модель также дала неоднозначный ответ для чанка {chunk.chunk_id}. Пробую еще более строгие настройки.")
                        self.notify_observers("status", {"message": self.translator.translate("retrieval_strict_attempt").format(chunk_id=chunk.chunk_id)})

                        # Попытка 3: Запасная модель с максимально детерминированными настройками
                        # Переопределяем параметры генерации прямо здесь
                        prompt_attempt3 = self.config.retriever_fallback_prompt.format( # Можно использовать тот же промпт
                            prompt=rag_query.question,
                            chunk_content=chunk.content
                        )
                        messages_attempt3 = [{"role": "user", "content": prompt_attempt3}]

                        llm_response_strict = self.fallback_inference_engine.generate(
                            messages=messages_attempt3,
                            # Очень строгие параметры:
                            temperature=0.01, # Почти детерминированный
                            max_new_tokens=2, # Ожидаем только "да" или "нет"
                            top_p=0.1,
                            top_k=1,
                            repeat_penalty=1.0,
                            stop=["\n", ".", ",", "!", "?", "Да.", "Нет."] # Здесь можно использовать стопы
                        )
                        processed_llm_response_strict = self._extract_yes_no(llm_response_strict)
                        self.logger.debug(f"Ответ LLM (попытка 3, strict) для чанка {chunk.chunk_id} (сырой): '{llm_response_strict.strip()}', обработанный: '{processed_llm_response_strict}'")
                        
                        # Используем результат самой строгой попытки
                        processed_llm_response = processed_llm_response_strict
                    else:
                        # Если запасная модель дала четкий ответ, используем его
                        processed_llm_response = processed_llm_response_fallback
                    
                    # Выгружаем запасную модель после использования
                    self.fallback_inference_engine.unload_model()
                else:
                    # Если основная модель дала четкий ответ, используем его
                    pass # processed_llm_response уже установлен

                # Оценка релевантности на основе лучшего результата
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
                        "llm_relevance_assessment": llm_response_attempt1.strip(), # Сохраняем ответ первой модели
                        "processed_llm_assessment": processed_llm_response, # Сохраняем финальный обработанный ответ
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
                    # Если после всех попыток ответ все еще не "да" или "нет"
                    self.logger.warning(f"Чанк {chunk.chunk_id}: Не удалось однозначно определить релевантность после всех попыток. Чанк пропущен.")
                    self.notify_observers("status", {"message": self.translator.translate("retrieval_undefined_relevance").format(chunk_id=chunk.chunk_id)})

                # Этот блок try/except должен быть внутри цикла for,
                # но не должен включать unload_model()
                # except Exception as e:
                #     self.logger.error(f"Ошибка при оценке релевантности чанка {chunk.chunk_id} с помощью LLM: {e}", exc_info=True)
                #     self.notify_observers("error", {"stage": "retrieval", "error": f"Ошибка LLM для чанка {chunk.chunk_id}: {str(e)}"})
                #     continue

        except Exception as e: # Этот общий except-блок для ошибок всего процесса retrieve
            self.logger.error(f"Критическая ошибка в процессе ретривинга: {e}", exc_info=True)
            self.notify_observers("error", {"stage": "retrieval", "error": f"Критическая ошибка ретривинга: {str(e)}"})
        finally:
            # Выгружаем основную модель после завершения ретривинга (или при ошибке)
            inference_engine.unload_model()

        self.logger.info(f"Ретривинг завершен. Найдено {len(relevant_chunks)} релевантных чанков.")
        self.notify_observers("complete", {"stage": "retrieval", "relevant_chunks_count": len(relevant_chunks)})
        return relevant_chunks

    def _extract_yes_no(self, text: str) -> str:
        """
        Извлекает 'да' или 'нет' из ответа LLM, используя не строгие паттерны.
        """
        text_lower = text.strip().lower()
        if self.yes_pattern.search(text_lower):
            return "да"
        if self.no_pattern.search(text_lower):
            return "нет"
        return "" # Возвращаем пустую строку, если не удалось однозначно определить

