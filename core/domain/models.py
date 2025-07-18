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
                f"file_name='{self.file_path.name}', "
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

class InferenceConfig:
    """
    DTO для конфигурации инференс-движка и параметров генерации.
    """
    def __init__(self,
                 engine_type: Literal["llamacpp", "vllm_stub", "hf_transformers_stub"] = "llamacpp",
                 model_path: Path = Path("models/default_model.gguf"),
                 n_gpu_layers: int = 0, # Количество слоев, выгружаемых на GPU (для llama.cpp)
                 device_type: Literal["cpu", "cuda", "amd", "integrated", "auto"] = "auto", # Тип устройства
                 n_ctx: int = 2048, # Размер контекстного окна для модели
                 temperature: float = 0.7,
                 max_new_tokens: int = 500,
                 top_p: float = 0.95,
                 top_k: int = 40,
                 repeat_penalty: float = 1.1,
                 stop_sequences: Optional[List[str]] = None,
                 prompt_template: str = "{prompt}"): # Общий шаблон промпта
        self.engine_type = engine_type
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.device_type = device_type
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.stop_sequences = stop_sequences if stop_sequences is not None else []
        self.prompt_template = prompt_template

class RetrievalConfig:
    """
    DTO для конфигурации модуля ретривинга.
    """
    def __init__(self,
                 strategy_type: int = 1, # Номер стратегии (для фабрики)
                 top_k: int = 5, # Сколько наиболее релевантных чанков вернуть
                 keywords: Optional[List[str]] = None, # Для стратегии с ключевыми словами
                 retriever_prompt: str = ""): # ИЗМЕНЕНО: Добавляем retriever_prompt
        self.strategy_type = strategy_type
        self.top_k = top_k
        self.keywords = keywords if keywords is not None else []
        self.retriever_prompt = retriever_prompt # ИЗМЕНЕНО: Инициализируем retriever_prompt

class SynthesisConfig:
    """
    DTO для конфигурации модуля синтеза/генерации ответа.
    """
    def __init__(self,
                 prompt_template: str = ""): # Специфический шаблон промпта для синтеза
        self.prompt_template = prompt_template

class RAGConfig:
    """
    Общий DTO для всех конфигураций RAG.
    """
    def __init__(self,
                 chunking_config: ChunkingConfig,
                 retrieval_config: RetrievalConfig,
                 synthesis_config: SynthesisConfig,
                 retrieval_inference_config: InferenceConfig, # Конфиг для инференса ретривера
                 synthesis_inference_config: InferenceConfig, # Конфиг для инференса синтеза
                 general_language: str = 'en'): # Язык для промптов и т.д.
        self.chunking = chunking_config
        self.retrieval = retrieval_config
        self.synthesis = synthesis_config
        self.retrieval_inference = retrieval_inference_config
        self.synthesis_inference = synthesis_inference_config
        self.general_language = general_language
