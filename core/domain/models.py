# core/domain/models.py
from pathlib import Path
from typing import List, Optional, Literal, Dict

class RAGQuery:
    """
    DTO для инкапсуляции пользовательского запроса к системе RAG.
    """
    def __init__(self,
                 question: str,
                 input_path: Path, # Может быть путем к файлу или папке
                 output_dir: Path):
        self.question = question
        self.input_path = input_path
        self.output_dir = output_dir

    def __repr__(self):
        return (f"RAGQuery(question='{self.question}', "
                f"input_path='{self.input_path}', "
                f"output_dir='{self.output_dir}')")

class Chunk:
    """
    DTO для представления одного фрагмента текста (чанка).
    """
    def __init__(self,
                 content: str,
                 file_path: Path,
                 chunk_id: str,
                 start_offset: int,
                 end_offset: int,
                 metadata: Optional[dict] = None):
        self.content = content
        self.file_path = file_path
        self.chunk_id = chunk_id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return (f"Chunk(chunk_id='{self.chunk_id}', "
                f"file_path='{self.file_path.name}', "
                f"start={self.start_offset}, end={self.end_offset}, "
                f"content='{self.content[:50]}...')")

class ChunkingConfig:
    """
    DTO для конфигурации модуля чанкинга.
    """
    def __init__(self,
                 chunk_size: int = 2000,
                 overlap_size: int = 200,
                 chunk_by: Literal["characters", "sentences", "paragraphs", "recursive"] = "recursive",
                 keep_sentences_together: bool = True,
                 encoding: str = 'utf-8',
                 language: str = 'en',
                 min_chunk_size: int = 100,
                 model_name: Optional[str] = None):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.chunk_by = chunk_by
        self.keep_sentences_together = keep_sentences_together
        self.encoding = encoding
        self.language = language
        self.min_chunk_size = min_chunk_size
        self.model_name = model_name

class RetrievalConfig:
    """
    DTO для конфигурации модуля ретривинга.
    """
    def __init__(self,
                 strategy_type: int = 1, # Номер стратегии (для фабрики)
                 model_path: Optional[Path] = None, # Путь к модели для ретривинга
                 keywords: Optional[List[str]] = None, # Для стратегии с ключевыми словами
                 top_k: int = 5): # Сколько наиболее релевантных чанков вернуть
        self.strategy_type = strategy_type
        self.model_path = model_path
        self.keywords = keywords if keywords is not None else []
        self.top_k = top_k

class SynthesisConfig:
    """
    DTO для конфигурации модуля синтеза/генерации ответа.
    """
    def __init__(self,
                 model_path: Optional[Path] = None, # Путь к финальной модели
                 temperature: float = 0.7,
                 max_new_tokens: int = 500,
                 prompt_template: str = ""):
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template # Возможно, специфичный промпт

class RAGConfig:
    """
    Общий DTO для всех конфигураций RAG.
    """
    def __init__(self,
                 chunking_config: ChunkingConfig,
                 retrieval_config: RetrievalConfig,
                 synthesis_config: SynthesisConfig,
                 general_language: str = 'en'): # Язык для промптов и т.д.
        self.chunking = chunking_config
        self.retrieval = retrieval_config
        self.synthesis = synthesis_config
        self.general_language = general_language

# Добавьте другие DTO по мере необходимости, например, RetrievalResult, Answer