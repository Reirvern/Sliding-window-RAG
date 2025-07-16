# core/factories/chunker_factory.py
import logging
from pathlib import Path
from typing import Dict, Type

from core.chunking.base_chunker import BaseChunker
from core.chunking.text_chunker import TextChunker # Импортируем текстовый чанкер
from core.domain.models import ChunkingConfig

class ChunkerFactory:
    """
    Фабрика для создания экземпляров чанкеров на основе типа файла.
    """
    # Словарь для регистрации чанкеров по расширениям файлов
    _chunker_map: Dict[str, Type[BaseChunker]] = {
        ".txt": TextChunker,
        # TODO: Добавьте сюда другие типы файлов и соответствующие чанкеры
        # ".pdf": PdfChunker,
        # ".docx": DocxChunker,
    }

    @classmethod
    def get_chunker(cls, file_path: Path, config: ChunkingConfig, logger: logging.Logger) -> BaseChunker:
        """
        Возвращает экземпляр чанкера для заданного типа файла.
        
        Args:
            file_path: Путь к файлу.
            config: Конфигурация чанкинга.
            logger: Логгер.
            
        Returns:
            Экземпляр BaseChunker или его подкласса.
            
        Raises:
            ValueError: Если тип файла не поддерживается.
        """
        file_extension = file_path.suffix.lower()
        chunker_class = cls._chunker_map.get(file_extension)

        if not chunker_class:
            logger.warning(f"Чанкер для расширения '{file_extension}' не найден. Использование текстового чанкера по умолчанию.")
            # Можно вернуть TextChunker по умолчанию или выкинуть ошибку
            chunker_class = TextChunker # Использование TextChunker как запасной вариант

            # Если хотите строгость, можно раскомментировать:
            # raise ValueError(f"Неподдерживаемое расширение файла: {file_extension}")
        
        return chunker_class(config, logger)