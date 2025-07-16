# core/chunking/base_chunker.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Callable, Any
import logging

from core.domain.models import Chunk, ChunkingConfig

class BaseChunker(ABC):
    """
    Абстрактный базовый класс для всех чанкеров.
    Определяет общий интерфейс для нарезки файлов на чанки.
    """
    def __init__(self, config: ChunkingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def chunk_file(self,
                   file_path: Path,
                   output_dir: Path,
                   file_index: int, # Добавил для уникальности имен файлов чанков
                   progress_callback: Callable[[int, int], None]) -> List[Chunk]:
        """
        Нарезает один файл на чанки и сохраняет их.
        
        Args:
            file_path: Путь к файлу для обработки.
            output_dir: Директория для сохранения чанков.
            file_index: Индекс текущего файла в списке (для уникальности имени файла).
            progress_callback: Функция обратного вызова для обновления прогресса.
                               Принимает (текущий_чанк, всего_чанков_в_файле).

        Returns:
            Список объектов Chunk.
        """
        pass