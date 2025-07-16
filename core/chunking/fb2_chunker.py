# core/chunking/fb2_chunker.py
import logging
import re
from pathlib import Path
from typing import List, Callable, Dict, Any
import json
import xml.etree.ElementTree as ET # Импортируем XML парсер

from core.chunking.base_chunker import BaseChunker
from core.domain.models import Chunk, ChunkingConfig

# Пространства имен FB2 для корректного парсинга XML
# (могут меняться, но это наиболее распространенные)
FB2_NAMESPACES = {
    'f': 'http://www.gribuser.ru/xml/fictionbook/2.0',
    'l': 'http://www.w3.org/1999/xlink'
}

class FB2Chunker(BaseChunker):
    """
    Чанкер для файлов формата FictionBook (FB2).
    Извлекает текстовое содержимое и нарезает его на чанки.
    """
    def __init__(self, config: ChunkingConfig, logger: logging.Logger):
        super().__init__(config, logger)
        self.chunk_id_counter = 0 # Счетчик для уникальных ID чанков

    def chunk_file(self,
                   file_path: Path,
                   output_dir: Path,
                   file_index: int,
                   progress_callback: Callable[[int, int], None]) -> List[Chunk]:
        """
        Нарезает FB2 файл на чанки.
        """
        self.logger.info(f"Чанкинг FB2 файла: {file_path.name}")
        chunks: List[Chunk] = []
        full_text_content = ""

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Извлекаем текст из основных разделов книги
            # Ищем теги <body f:name="main"> или просто <body>
            main_body = root.find('f:body[@f:name="main"]', FB2_NAMESPACES) or root.find('f:body', FB2_NAMESPACES)

            if main_body is None:
                self.logger.warning(f"Не найден основной 'body' в файле {file_path.name}. Попытка извлечь весь текст.")
                full_text_content = self._extract_text_from_element(root)
            else:
                full_text_content = self._extract_text_from_element(main_body)

        except ET.ParseError as e:
            self.logger.error(f"Ошибка парсинга FB2 файла {file_path.name}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Не удалось прочитать или обработать FB2 файл {file_path.name}: {e}", exc_info=True)
            return []

        if not full_text_content.strip():
            self.logger.warning(f"FB2 файл {file_path.name} пуст или не содержит извлекаемого текста. Чанкинг пропущен.")
            return []

        # Выбираем метод разбиения, аналогично TextChunker
        if self.config.chunk_by == "characters":
            raw_chunks = self._chunk_by_characters(full_text_content)
        elif self.config.chunk_by == "sentences":
            raw_chunks = self._chunk_by_sentences(full_text_content)
        elif self.config.chunk_by == "paragraphs":
            raw_chunks = self._chunk_by_paragraphs(full_text_content)
        else:
            self.logger.warning(f"Неизвестный тип разбиения '{self.config.chunk_by}'. Использую разбиение по символам.")
            raw_chunks = self._chunk_by_characters(full_text_content)

        # Применяем overlap и post-processing (min_chunk_size)
        processed_chunks = self._apply_overlap_and_min_size(raw_chunks)
        
        total_chunks_in_file = len(processed_chunks)
        for i, (content, start_offset, end_offset) in enumerate(processed_chunks):
            self.chunk_id_counter += 1
            chunk_id = f"chunk_{file_index}_{self.chunk_id_counter:05d}"
            
            chunk = Chunk(
                content=content,
                file_path=file_path,
                chunk_id=chunk_id,
                start_offset=start_offset, # Эти смещения будут относительными к извлеченному full_text_content
                end_offset=end_offset,     # а не к исходному XML. Это упрощение.
                metadata={
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "chunk_source_type": self.config.chunk_by,
                    "language": self.config.language,
                    "original_format": "fb2"
                }
            )
            chunks.append(chunk)
            self._save_chunk_to_json(chunk, output_dir)
            progress_callback(i + 1, total_chunks_in_file)

        # ИЗМЕНЕНО: Удален DEBUG лог, который мог мешать tqdm
        # self.logger.debug(f"Создано {len(chunks)} чанков из файла {file.name}.")
        return chunks

    def _extract_text_from_element(self, element: ET.Element) -> str:
        """
        Рекурсивно извлекает весь текстовый контент из XML-элемента,
        добавляя переносы строк для разделения абзацев/блоков.
        """
        texts = []
        if element.text:
            texts.append(element.text.strip())

        for child in element:
            # Обрабатываем специфические теги для форматирования
            tag_name = child.tag.split('}')[-1] # Получаем имя тега без пространства имен

            if tag_name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'subtitle', 'text-author', 'epigraph', 'annotation', 'poem', 'stanza', 'cite']:
                # Добавляем двойной перенос строки для абзацев и заголовков
                texts.append(self._extract_text_from_element(child) + "\n\n")
            elif tag_name in ['strong', 'emphasis', 'strikethrough', 'sub', 'sup']:
                # Встроенные теги, просто извлекаем текст без дополнительных переносов
                texts.append(self._extract_text_from_element(child))
            else:
                # Для остальных тегов, просто рекурсивно извлекаем текст
                texts.append(self._extract_text_from_element(child))

            if child.tail:
                texts.append(child.tail.strip())
        
        return " ".join(filter(None, texts)).strip() # Объединяем, убирая пустые строки

    # Методы _chunk_by_characters, _chunk_by_sentences, _chunk_by_paragraphs,
    # _apply_overlap_and_min_size и _save_chunk_to_json
    # могут быть скопированы из TextChunker.py, так как они работают с обычным текстом.
    # Для избежания дублирования кода, в будущем их можно вынести в отдельный
    # вспомогательный модуль или сделать TextChunker базовым для других текстовых чанкеров.

    # Для текущей задачи, я скопирую их сюда, чтобы FB2Chunker был самодостаточным.
    # В реальном проекте, я бы рекомендовал рефакторинг этих общих методов.

    def _chunk_by_characters(self, text: str) -> List[tuple[str, int, int]]:
        """Разбивает текст по символам с учетом chunk_size и overlap_size."""
        results = []
        current_offset = 0
        while current_offset < len(text):
            end_offset = min(current_offset + self.config.chunk_size, len(text))
            chunk_content = text[current_offset:end_offset]
            results.append((chunk_content, current_offset, end_offset))
            current_offset += self.config.chunk_size - self.config.overlap_size
            if self.config.overlap_size > self.config.chunk_size: # Избежать бесконечного цикла
                self.logger.warning("Overlap size cannot be greater than chunk size. Adjusting overlap.")
                self.config.overlap_size = self.config.chunk_size // 2
        return results

    def _chunk_by_sentences(self, text: str) -> List[tuple[str, int, int]]:
        """
        Разбивает текст по предложениям.
        Простая реализация: ищет точки, восклицательные/вопросительные знаки.
        Для более точного разбиения по предложениям требуются библиотеки, такие как NLTK.
        """
        # Разделители предложений: ., !, ? (с учетом кавычек)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        results = []
        current_offset = 0
        current_chunk_content = ""
        current_chunk_start_offset = 0

        for sentence in sentences:
            sentence_len = len(sentence) + (1 if sentence and sentence[-1] in ".!?" else 2) # Учитываем пробел после
            
            # Если добавление предложения превысит chunk_size, сохраняем текущий чанк
            if len(current_chunk_content) + sentence_len > self.config.chunk_size and current_chunk_content:
                results.append((current_chunk_content.strip(), current_chunk_start_offset, current_offset))
                current_chunk_content = ""
                # Перекрываем, если нужно
                current_chunk_start_offset = max(0, current_offset - self.config.overlap_size)
                
            if not current_chunk_content: # Если чанк пуст, устанавливаем новый старт
                current_chunk_start_offset = current_offset

            current_chunk_content += sentence + " " # Добавляем предложение и пробел
            current_offset += sentence_len

        if current_chunk_content: # Добавляем последний чанк
            results.append((current_chunk_content.strip(), current_chunk_start_offset, current_offset))
        
        return results

    def _chunk_by_paragraphs(self, text: str) -> List[tuple[str, int, int]]:
        """
        Разбивает текст по абзацам.
        Абзацы определяются как текст, разделенный двумя или более переносами строки.
        """
        paragraphs = re.split(r'\n\s*\n+', text.strip()) # Разбиваем по двойным переносам строк
        
        results = []
        current_offset = 0
        
        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            # Ищем точное смещение абзаца в исходном тексте
            # Это может быть не идеально, если есть много пробелов или спецсимволов
            para_start_offset = text.find(para_stripped, current_offset)
            if para_start_offset == -1: # Если не нашли, ищем с начала
                para_start_offset = text.find(para_stripped)
            
            para_end_offset = para_start_offset + len(para_stripped)
            
            # Если абзац слишком большой, разбиваем его на символы
            if len(para_stripped) > self.config.chunk_size:
                self.logger.warning(f"Параграф слишком большой ({len(para_stripped)} символов), будет разбит по символам.")
                # Рекурсивно разбиваем большой абзац по символам
                sub_chunks = self._chunk_by_characters(para_stripped)
                for sub_content, sub_start, sub_end in sub_chunks:
                    results.append((sub_content, para_start_offset + sub_start, para_start_offset + sub_end))
            else:
                results.append((para_stripped, para_start_offset, para_end_offset))
            
            current_offset = para_end_offset # Обновляем текущее смещение для поиска следующего абзаца

        # Теперь объединим абзацы в чанки, если они слишком маленькие или нужно сделать overlap
        final_chunks = []
        current_chunk_content = ""
        current_chunk_start_offset = 0

        if not results: return []

        for content, start, end in results:
            if not current_chunk_content:
                current_chunk_start_offset = start
            
            # Проверяем, не превысит ли добавление текущего контента chunk_size
            # Если превысит и текущий чанк уже не пуст, сохраняем его
            if len(current_chunk_content) + len(content) + 1 > self.config.chunk_size and current_chunk_content:
                final_chunks.append((current_chunk_content.strip(), current_chunk_start_offset, end)) 
                current_chunk_content = ""
                # Начинаем новый чанк с перекрытием, если это возможно
                current_chunk_start_offset = max(0, start - self.config.overlap_size) 

            if not current_chunk_content: # Для нового чанка
                current_chunk_start_offset = start

            current_chunk_content += content + "\n\n" # Добавляем абзац и двойной перенос строки

        if current_chunk_content:
            final_chunks.append((current_chunk_content.strip(), current_chunk_start_offset, end)) # Последний чанк
        
        return final_chunks


    def _apply_overlap_and_min_size(self, raw_chunks: List[tuple[str, int, int]]) -> List[tuple[str, int, int]]:
        """Применяет перекрытие и объединение чанков по min_chunk_size."""
        processed_chunks = []
        if not raw_chunks:
            return []

        # Объединение слишком маленьких чанков
        temp_chunks = []
        current_combined_chunk = ""
        current_start_offset = raw_chunks[0][1] # Начальное смещение первого чанка

        for i, (content, start, end) in enumerate(raw_chunks):
            # Если текущий чанк слишком мал ИЛИ мы не последний чанк (есть с чем объединять)
            # пытаемся объединить его с текущим накопленным чанком
            if len(content) < self.config.min_chunk_size and (current_combined_chunk or i < len(raw_chunks) - 1):
                if not current_combined_chunk:
                    current_start_offset = start # Если начинаем новый комбинированный чанк
                current_combined_chunk += content + " " # Добавляем содержимое
                
            else: # Чанк достаточно большой или это последний маленький чанк
                if current_combined_chunk: # Если есть накопленный маленький чанк, добавляем его
                    current_combined_chunk += content + " " # Добавляем и текущий (возможно, большой)
                    # Конец комбинированного чанка - это конец текущего, если он не пустой
                    combined_end_offset = end if content else (start + len(current_combined_chunk.strip()))
                    temp_chunks.append((current_combined_chunk.strip(), current_start_offset, combined_end_offset))
                    current_combined_chunk = "" # Сбрасываем
                else: # Если накопленного нет, добавляем текущий как есть
                    temp_chunks.append((content, start, end))
                current_start_offset = start # Обновляем старт для следующего

        # Повторно проверяем последний накопленный чанк, если он остался
        if current_combined_chunk:
            # Если остался только маленький комбинированный чанк и он не соответствует min_chunk_size,
            # и если есть предыдущий чанк, можно попробовать объединить с ним
            if temp_chunks:
                last_chunk_content, last_chunk_start, last_chunk_end = temp_chunks[-1]
                # Объединяем, если последний чанк тоже мал и не превысит chunk_size
                if len(current_combined_chunk.strip()) < self.config.min_chunk_size and \
                   len(last_chunk_content) + len(current_combined_chunk.strip()) + 1 <= self.config.chunk_size:
                    temp_chunks[-1] = (last_chunk_content + " " + current_combined_chunk.strip(), last_chunk_start, last_chunk_end + len(current_combined_chunk.strip()))
                else: # Иначе добавляем как отдельный
                    temp_chunks.append((current_combined_chunk.strip(), current_start_offset, current_start_offset + len(current_combined_chunk.strip())))
            else: # Если это единственный чанк
                temp_chunks.append((current_combined_chunk.strip(), current_start_offset, current_start_offset + len(current_combined_chunk.strip())))

        # Дополнительная проверка на min_chunk_size после всех операций
        final_processed_chunks = []
        for content, start, end in temp_chunks:
            if len(content) >= self.config.min_chunk_size:
                final_processed_chunks.append((content, start, end))
            else:
                self.logger.warning(f"Чанк [{content[:50]}...] имеет размер {len(content)} < min_chunk_size {self.config.min_chunk_size}. Он может быть пропущен или объединен.")
                if len(content) > 0: # Не добавляем совсем пустые чанки
                    final_processed_chunks.append((content, start, end))

        return final_processed_chunks


    def _save_chunk_to_json(self, chunk: Chunk, output_dir: Path):
        """Сохраняет один чанк в JSON файл."""
        chunk_data = {
            "chunk_id": chunk.chunk_id,
            "file_name": chunk.file_path.name,
            "original_file_path": str(chunk.file_path),
            "content": chunk.content,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "length": len(chunk.content),
            "metadata": chunk.metadata
        }
        
        # Убедимся, что папка для чанков существует
        chunks_output_path = output_dir / "chunks"
        chunks_output_path.mkdir(parents=True, exist_ok=True)

        # Сохраняем чанк в файл
        chunk_file_name = f"{chunk.chunk_id}.json"
        chunk_file_path = chunks_output_path / chunk_file_name
        try:
            with open(chunk_file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=4)
            # ИЗМЕНЕНО: УДАЛЕН DEBUG лог для каждого сохраненного чанка
            # self.logger.debug(f"Чанк {chunk.chunk_id} сохранен в {chunk_file_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении чанка {chunk.chunk_id} в {chunk_file_path}: {e}")

