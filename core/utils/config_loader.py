# core/utils/config_loader.py
import json
from pathlib import Path
from core.domain.models import RAGConfig, ChunkingConfig, RetrievalConfig, SynthesisConfig, InferenceConfig
import logging # Импортируем логгер

# Получаем логгер AltRAG, чтобы использовать его для отладочных сообщений
logger = logging.getLogger('AltRAG')

def load_config(config_path: Path) -> dict:
    """Загружает общую конфигурацию приложения из JSON файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.critical(f"Файл конфигурации не найден: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Ошибка парсинга JSON в файле {config_path}: {e}")
        raise

def load_rag_config(config_path: Path) -> RAGConfig:
    """Загружает конфигурацию RAG из JSON файла."""
    logger.info(f"Загружаю RAG конфигурацию из: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Сырые данные RAG конфига из файла: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except FileNotFoundError:
        logger.critical(f"Файл RAG конфигурации не найден: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Ошибка парсинга JSON в файле RAG конфига {config_path}: {e}")
        raise

    chunking_conf_data = data.get("chunking", {})
    chunking_config = ChunkingConfig(
        chunk_size=chunking_conf_data.get("chunk_size", 2000),
        overlap_size=chunking_conf_data.get("overlap_size", 200),
        chunk_by=chunking_conf_data.get("chunk_by", "recursive"),
        keep_sentences_together=chunking_conf_data.get("keep_sentences_together", True),
        encoding=chunking_conf_data.get("encoding", "utf-8"),
        language=chunking_conf_data.get("language", "en"),
        min_chunk_size=chunking_conf_data.get("min_chunk_size", 100),
        model_name=chunking_conf_data.get("model_name")
    )
    logger.debug(f"ChunkingConfig создан: chunk_size={chunking_config.chunk_size}, chunk_by={chunking_config.chunk_by}")

    retrieval_conf_data = data.get("retrieval", {})
    retrieval_config = RetrievalConfig(
        strategy_type=retrieval_conf_data.get("strategy_type", 1),
        keywords=retrieval_conf_data.get("keywords"),
        top_k=retrieval_conf_data.get("top_k", 5)
    )
    logger.debug(f"RetrievalConfig создан: strategy_type={retrieval_config.strategy_type}, top_k={retrieval_config.top_k}")

    synthesis_conf_data = data.get("synthesis", {})
    synthesis_config = SynthesisConfig(
        prompt_template=synthesis_conf_data.get("prompt_template", "")
    )
    logger.debug(f"SynthesisConfig создан: prompt_template_len={len(synthesis_config.prompt_template)}")
    
    # Загрузка конфигурации для инференса ретривера
    retrieval_inference_conf_data = data.get("retrieval_inference", {})
    logger.debug(f"Сырые данные retrieval_inference: {retrieval_inference_conf_data}")
    retrieval_inference_config = InferenceConfig(
        engine_type=retrieval_inference_conf_data.get("engine_type", "llamacpp"),
        model_path=Path(retrieval_inference_conf_data.get("model_path", "models/default_retrieval_model.gguf")),
        n_gpu_layers=retrieval_inference_conf_data.get("n_gpu_layers", 0),
        device_type=retrieval_inference_conf_data.get("device_type", "auto"),
        n_ctx=retrieval_inference_conf_data.get("n_ctx", 2048),
        temperature=retrieval_inference_conf_data.get("temperature", 0.1),
        max_new_tokens=retrieval_inference_conf_data.get("max_new_tokens", 50),
        top_p=retrieval_inference_conf_data.get("top_p", 0.95),
        top_k=retrieval_inference_conf_data.get("top_k", 40),
        repeat_penalty=retrieval_inference_conf_data.get("repeat_penalty", 1.1),
        stop_sequences=retrieval_inference_conf_data.get("stop_sequences", ["\n"]),
        prompt_template=retrieval_inference_conf_data.get("prompt_template", "{prompt}")
    )
    logger.debug(f"RetrievalInferenceConfig создан: n_ctx={retrieval_inference_config.n_ctx}, model_path={retrieval_inference_config.model_path.name}")


    # Загрузка конфигурации для инференса синтеза
    synthesis_inference_conf_data = data.get("synthesis_inference", {})
    logger.debug(f"Сырые данные synthesis_inference: {synthesis_inference_conf_data}")
    synthesis_inference_config = InferenceConfig(
        engine_type=synthesis_inference_conf_data.get("engine_type", "llamacpp"),
        model_path=Path(synthesis_inference_conf_data.get("model_path", "models/default_synthesis_model.gguf")),
        n_gpu_layers=synthesis_inference_conf_data.get("n_gpu_layers", 0),
        device_type=synthesis_inference_conf_data.get("device_type", "auto"),
        n_ctx=synthesis_inference_conf_data.get("n_ctx", 2048),
        temperature=synthesis_inference_conf_data.get("temperature", 0.7),
        max_new_tokens=synthesis_inference_conf_data.get("max_new_tokens", 500),
        top_p=synthesis_inference_conf_data.get("top_p", 0.95),
        top_k=synthesis_inference_conf_data.get("top_k", 40),
        repeat_penalty=synthesis_inference_conf_data.get("repeat_penalty", 1.1),
        stop_sequences=synthesis_inference_conf_data.get("stop_sequences", ["\n", "Вопрос:"]),
        prompt_template=synthesis_inference_conf_data.get("prompt_template", "{prompt}")
    )
    logger.debug(f"SynthesisInferenceConfig создан: n_ctx={synthesis_inference_config.n_ctx}, model_path={synthesis_inference_config.model_path.name}")


    return RAGConfig(
        chunking_config=chunking_config,
        retrieval_config=retrieval_config,
        synthesis_config=synthesis_config,
        retrieval_inference_config=retrieval_inference_config,
        synthesis_inference_config=synthesis_inference_config,
        general_language=data.get("general_language", "en")
    )
