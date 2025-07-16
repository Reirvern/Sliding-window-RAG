# core/services/chunking_service.py
import logging
from pathlib import Path
from typing import List
from core.domain.models import RAGQuery, ChunkingConfig, Chunk
from core.utils.observer import Observable
from core.factories.chunker_factory import ChunkerFactory # Импортируем фабрику

class ChunkingService(Observable):
    """
    Сервис для нарезки входных документов на чанки.
    """
    def __init__(self, config: ChunkingConfig, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.chunker_factory = ChunkerFactory() # Инициализируем фабрику чанкеров

    def process(self, rag_query: RAGQuery) -> List[Chunk]:
        """
        Обрабатывает входные файлы и нарезает их на чанки.
        """
        self.logger.info(f"Начинаю чанкинг для {rag_query.input_path} с конфигом: {self.config.chunk_by}, size={self.config.chunk_size}")
        
        all_chunks: List[Chunk] = []
        input_files = self._get_files_from_path(rag_query.input_path)
        
        total_files = len(input_files)
        if total_files == 0:
            self.logger.warning(f"Входная директория/файл {rag_query.input_path} не содержит обрабатываемых файлов.")
            self.notify_observers("complete", {"stage": "chunking", "total_chunks": 0})
            return []

        for i, file_path in enumerate(input_files):
            try:
                # Получаем соответствующий чанкер через фабрику
                chunker = self.chunker_factory.get_chunker(file_path, self.config, self.logger)
                
                # Функция обратного вызова для прогресса в рамках одного файла
                def file_progress_callback(current_chunk_in_file: int, total_chunks_in_file: int):
                    # Преобразуем прогресс файла в общий прогресс чанкинга
                    # Это упрощенный расчет, можно сделать более точный, если известны размеры файлов
                    overall_progress = (i / total_files) + (current_chunk_in_file / total_chunks_in_file / total_files)
                    self.notify_observers("progress", {
                        "stage": "chunking",
                        "current": i + 1, # Номер текущего файла
                        "total": total_files, # Общее количество файлов
                        "file_name": file_path.name,
                        "file_progress_percent": int(overall_progress * 100) # Прогресс в % по всем файлам
                    })

                file_chunks = chunker.chunk_file(file_path, rag_query.output_dir, i, file_progress_callback)
                all_chunks.extend(file_chunks)
                self.logger.debug(f"Обработан файл {file_path.name}, создано {len(file_chunks)} чанков.")
            except ValueError as e:
                self.logger.error(f"Ошибка при чанкинге файла {file_path.name}: {e}")
            except Exception as e:
                self.logger.error(f"Непредвиденная ошибка при чанкинге файла {file_path.name}: {e}", exc_info=True)


        self.logger.info(f"Чанкинг завершен. Создано всего {len(all_chunks)} чанков.")
        self.notify_observers("complete", {"stage": "chunking", "total_chunks": len(all_chunks)})
        return all_chunks

    def _get_files_from_path(self, path: Path) -> List[Path]:
        """Вспомогательный метод для получения списка файлов из пути."""
        if path.is_file():
            return [path]
        elif path.is_dir():
            # TODO: Отфильтровать по поддерживаемым типам файлов, определенным в ChunkerFactory._chunker_map.keys()
            supported_extensions = ChunkerFactory._chunker_map.keys()
            files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
            return files
        return []