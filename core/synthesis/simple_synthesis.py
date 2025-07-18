# core/synthesis/simple_synthesis.py
import logging
import json
from pathlib import Path
from typing import List, Any, Dict, Tuple
import re
from datetime import datetime 

from core.synthesis.base_synthesis import BaseSynthesis
from core.domain.models import RAGQuery, Chunk, SynthesisConfig, InferenceConfig, SynthesisResult
from core.utils.localization.translator import Translator

class SimpleSynthesis(BaseSynthesis):
    """
    Простая стратегия синтеза, которая объединяет релевантные чанки
    и генерирует ответ с помощью LLM, управляя контекстным окном.
    """
    def __init__(self,
                 config: SynthesisConfig,
                 inference_config: InferenceConfig,
                 logger: logging.Logger,
                 translator: Translator):
        super().__init__(config, inference_config, logger, translator)
        self.logger.debug("SimpleSynthesis инициализирован.")
        # Паттерн для извлечения цитат из ответа модели
        self.citation_pattern = re.compile(r'\[ЦИТАТА:\s*"(.*?)"\]', re.IGNORECASE)


    def synthesize(self,
                   rag_query: RAGQuery,
                   relevant_chunks: List[Chunk],
                   inference_engine: Any) -> SynthesisResult:
        """
        Синтезирует финальный ответ на основе релевантных чанков и запроса.
        Управляет контекстным окном модели и ищет цитаты.
        """
        self.logger.info("Начинаю процесс синтеза с SimpleSynthesis.")
        self.notify_observers("status", {"message": self.translator.translate("synthesis_started")})

        if not relevant_chunks:
            self.logger.warning("Нет релевантных чанков для синтеза. Возвращаю пустой ответ.")
            self.notify_observers("complete", {"stage": "synthesis", "answer": self.translator.translate("no_relevant_chunks_for_synthesis")})
            return SynthesisResult(answer=self.translator.translate("no_relevant_chunks_for_synthesis"), citations=[])

        inference_engine.load_model() 

        max_context_tokens = self.inference_config.n_ctx - self.config.context_token_buffer
        self.logger.debug(f"Максимальное количество токенов для контекста: {max_context_tokens} (n_ctx: {self.inference_config.n_ctx}, buffer: {self.config.context_token_buffer})")

        context_blocks: List[str] = []
        current_context_block_content: List[str] = []
        current_context_block_tokens = 0

        self.logger.info(f"Собираю контекстные блоки из {len(relevant_chunks)} релевантных чанков.")
        self.notify_observers("progress", {"stage": "synthesis", "current": 0, "total": len(relevant_chunks), "message": self.translator.translate("synthesis_collecting_context")})

        for i, chunk in enumerate(relevant_chunks):
            chunk_content = chunk.content
            chunk_tokens = inference_engine.get_token_count(chunk_content)
            self.logger.debug(f"Чанк {chunk.chunk_id} имеет {chunk_tokens} токенов.")

            if current_context_block_tokens + chunk_tokens + 50 > max_context_tokens and current_context_block_content:
                context_blocks.append("\n\n".join(current_context_block_content))
                self.logger.debug(f"Контекстный блок {len(context_blocks)} завершен ({current_context_block_tokens} токенов). Начинаю новый.")
                current_context_block_content = []
                current_context_block_tokens = 0
            
            current_context_block_content.append(chunk_content)
            current_context_block_tokens += chunk_tokens
            
            self.notify_observers("progress", {
                "stage": "synthesis",
                "current": i + 1,
                "total": len(relevant_chunks),
                "message": self.translator.translate("synthesis_collecting_context_progress", 
                                                    current=i+1, total=len(relevant_chunks))
            })

        if current_context_block_content:
            context_blocks.append("\n\n".join(current_context_block_content))
            self.logger.debug(f"Последний контекстный блок {len(context_blocks)} завершен ({current_context_block_tokens} токенов).")

        all_answers: List[str] = []
        all_citations: List[Dict[str, Any]] = [] # Будем собирать все найденные цитаты

        self.logger.info(f"Сформировано {len(context_blocks)} контекстных блоков для генерации.")
        self.notify_observers("progress", {"stage": "synthesis", "current": 0, "total": len(context_blocks), "message": self.translator.translate("synthesis_generating_answers")})

        for i, context_block in enumerate(context_blocks):
            self.logger.info(f"Генерация ответа для контекстного блока {i+1}/{len(context_blocks)}.")
            
            system_prompt = self.translator.translate("synthesis_system_prompt")

            formatted_user_content = self.config.synthesis_prompt.format(
                context=context_block,
                question=rag_query.question
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_content}
            ]
            self.logger.debug(f"Промпт для LLM (блок {i+1}): {messages}")

            try:
                answer_with_citations = inference_engine.generate( # Модель теперь должна возвращать ответ с цитатами
                    messages=messages,
                    temperature=self.inference_config.temperature,
                    max_new_tokens=self.inference_config.max_new_tokens,
                    top_p=self.inference_config.top_p,
                    top_k=self.inference_config.top_k,
                    repeat_penalty=self.inference_config.repeat_penalty,
                    stop=self.inference_config.stop_sequences
                )
                self.logger.info(f"Ответ для блока {i+1} сгенерирован (длина: {len(answer_with_citations)}).")

                # ИЗМЕНЕНИЕ: Парсинг цитат из ответа модели
                parsed_answer, extracted_citations = self._parse_answer_and_citations(answer_with_citations)
                all_answers.append(parsed_answer) # Сохраняем чистый ответ
                
                # Теперь ищем эти extracted_citations в исходных чанках
                found_citations_for_block = self._find_citations_in_chunks(extracted_citations, relevant_chunks)
                all_citations.extend(found_citations_for_block) # Добавляем найденные цитаты

            except Exception as e:
                self.logger.error(f"Ошибка при генерации ответа для блока {i+1}: {e}", exc_info=True)
                all_answers.append(self.translator.translate("synthesis_generation_error").format(error=str(e)))

            self.notify_observers("progress", {
                "stage": "synthesis",
                "current": i + 1,
                "total": len(context_blocks),
                "message": self.translator.translate("synthesis_generating_answers_progress", 
                                                    current=i+1, total=len(context_blocks))
            })

        final_combined_answer = "\n\n".join(all_answers)
        self.logger.info(f"Все ответы сгенерированы. Общая длина: {len(final_combined_answer)}.")

        self.logger.info(f"Поиск цитат завершен. Найдено {len(all_citations)} цитат.")
        self.notify_observers("complete", {"stage": "synthesis", "citations_count": len(all_citations)})

        output_data = {
            "answer": final_combined_answer,
            "citations": all_citations,
            "timestamp": datetime.now().isoformat(),
            "question": rag_query.question,
            "input_path": str(rag_query.input_path)
        }
        
        answer_output_dir = rag_query.output_dir / "answer"
        answer_output_dir.mkdir(parents=True, exist_ok=True)
        answer_file_path = answer_output_dir / "final_answer.json"

        try:
            with open(answer_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Финальный ответ и цитаты сохранены в {answer_file_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении финального ответа: {e}", exc_info=True)

        return SynthesisResult(answer=final_combined_answer, citations=all_citations)

    def _parse_answer_and_citations(self, llm_output: str) -> Tuple[str, List[str]]:
        """
        Парсит ответ LLM, извлекая основной текст ответа и цитаты.
        """
        extracted_citations: List[str] = []
        clean_answer = llm_output
        
        # Находим все совпадения паттерна цитаты
        matches = list(self.citation_pattern.finditer(llm_output))
        
        # Если есть цитаты, извлекаем их и удаляем из основного текста ответа
        if matches:
            # Сначала извлекаем цитаты
            for match in matches:
                citation_text = match.group(1)
                extracted_citations.append(citation_text.strip())
            
            # Затем удаляем их из основного текста ответа
            clean_answer = self.citation_pattern.sub('', llm_output).strip()
            
            # Удаляем лишние пробелы, которые могли появиться после удаления цитат
            clean_answer = re.sub(r'\s{2,}', ' ', clean_answer).strip()
            
        self.logger.debug(f"Извлеченные цитаты: {extracted_citations}")
        self.logger.debug(f"Очищенный ответ: {clean_answer[:100]}...")
        return clean_answer, extracted_citations

    def _find_citations_in_chunks(self, extracted_citations: List[str], relevant_chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Ищет извлеченные цитаты в релевантных чанках, используя более гибкий поиск.
        """
        found_citations: List[Dict[str, Any]] = []
        
        for ext_citation_text in extracted_citations:
            # Попробуем найти точное совпадение
            found_exact = False
            for chunk in relevant_chunks:
                if ext_citation_text.lower() in chunk.content.lower():
                    found_citations.append({
                        "text": ext_citation_text,
                        "source_file": chunk.file_path.name,
                        "chunk_id": chunk.chunk_id,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "metadata": chunk.metadata
                    })
                    found_exact = True
                    break
            
            if not found_exact:
                # Если точного совпадения нет, пробуем искать подстроки, уменьшая длину
                self.logger.debug(f"Точная цитата '{ext_citation_text[:50]}...' не найдена. Пробую нечеткий поиск.")
                
                # Разбиваем цитату на предложения и ищем их
                citation_sentences = re.split(r'(?<=[.!?])\s+', ext_citation_text)
                
                for cit_sentence in citation_sentences:
                    if not cit_sentence.strip():
                        continue
                    
                    # Пробуем найти предложения из цитаты
                    for chunk in relevant_chunks:
                        if cit_sentence.strip().lower() in chunk.content.lower():
                            # Если нашли предложение, добавляем его как цитату
                            found_citations.append({
                                "text": cit_sentence.strip(),
                                "source_file": chunk.file_path.name,
                                "chunk_id": chunk.chunk_id,
                                "start_offset": chunk.start_offset,
                                "end_offset": chunk.end_offset,
                                "metadata": chunk.metadata
                            })
                            # Можно добавить флаг, чтобы не дублировать цитаты, если одно предложение
                            # нашлось в нескольких чанках, или если оно уже было найдено
                            break # Переходим к следующему предложению из цитаты
            
        # Удаляем дубликаты цитат (если одно и то же предложение нашлось несколько раз)
        unique_citations = []
        seen_texts = set()
        for citation in found_citations:
            if citation["text"].lower() not in seen_texts:
                unique_citations.append(citation)
                seen_texts.add(citation["text"].lower())
        
        return unique_citations

