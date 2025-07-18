# core/utils/config_loader.py
import json
from pathlib import Path
from typing import Dict, Any, Optional # Добавляем Optional
from core.domain.models import RAGConfig, ChunkingConfig, RetrievalConfig, SynthesisConfig, InferenceConfig
import logging

logger = logging.getLogger('AltRAG')

def load_rag_config(config_path: Path) -> RAGConfig:
    """Загружает конфигурацию RAG из JSON файла."""
    logger.info(f"Загружаю RAG конфигурацию из: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Сырые данные RAG конфига из файла: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except FileNotFoundError:
        logger.critical(f"Файл конфигурации RAG не найден: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Ошибка декодирования JSON в файле RAG конфигурации {config_path}: {e}")
        raise

    chunking_conf_data = data.get("chunking", {})
    chunking_config = ChunkingConfig(
        chunk_size=chunking_conf_data.get("chunk_size", 1000), 
        overlap_size=chunking_conf_data.get("overlap_size", 100),
        chunk_by=chunking_conf_data.get("chunk_by", "sentences"), 
        keep_sentences_together=chunking_conf_data.get("keep_sentences_together", True),
        encoding=chunking_conf_data.get("encoding", "utf-8"),
        language=chunking_conf_data.get("language", "ru"), 
        min_chunk_size=chunking_conf_data.get("min_chunk_size", 50),
        model_name=chunking_conf_data.get("model_name")
    )
    logger.debug(f"ChunkingConfig создан: chunk_size={chunking_config.chunk_size}, chunk_by={chunking_config.chunk_by}")


    retrieval_conf_data = data.get("retrieval", {})
    retrieval_config = RetrievalConfig(
        strategy_type=retrieval_conf_data.get("strategy_type", 1),
        keywords=retrieval_conf_data.get("keywords"),
        top_k=retrieval_conf_data.get("top_k", 3), 
        retriever_prompt=retrieval_conf_data.get("retriever_prompt", ""),
        retriever_fallback_prompt=retrieval_conf_data.get("retriever_fallback_prompt", "") # НОВОЕ: Чтение fallback промпта
    )
    logger.debug(f"RetrievalConfig создан: strategy_type={retrieval_config.strategy_type}, top_k={retrieval_config.top_k}")


    synthesis_conf_data = data.get("synthesis", {})
    synthesis_config = SynthesisConfig(
        strategy_type=synthesis_conf_data.get("strategy_type", 1),
        synthesis_prompt=synthesis_conf_data.get("synthesis_prompt", "Используя следующий контекст:\n{context}\n\nОтветь на вопрос: {question}"),
        context_token_buffer=synthesis_conf_data.get("context_token_buffer", 2000)
    )
    logger.debug(f"SynthesisConfig создан: synthesis_prompt_len={len(synthesis_config.synthesis_prompt)}")
    
    # Загрузка конфигурации для инференса ретривера
    retrieval_inference_conf_data = data.get("retrieval_inference", {})
    logger.debug(f"Сырые данные retrieval_inference: {retrieval_inference_conf_data}")
    retrieval_inference_config = InferenceConfig(
        engine_type=retrieval_inference_conf_data.get("engine_type", "llamacpp"),
        model_path=Path(retrieval_inference_conf_data.get("model_path", "models/default_retrieval_model.gguf")),
        n_gpu_layers=retrieval_inference_conf_data.get("n_gpu_layers", 0),
        device_type=retrieval_inference_conf_data.get("device_type", "auto"),
        n_ctx=retrieval_inference_conf_data.get("n_ctx", 4096), 
        temperature=retrieval_inference_conf_data.get("temperature", 0.1), 
        max_new_tokens=retrieval_inference_conf_data.get("max_new_tokens", 50), 
        top_p=retrieval_inference_conf_data.get("top_p", 0.9), 
        top_k=retrieval_inference_conf_data.get("top_k", 20), 
        repeat_penalty=retrieval_inference_conf_data.get("repeat_penalty", 1.0), 
        stop_sequences=retrieval_inference_conf_data.get("stop_sequences", []) 
    )
    logger.debug(f"RetrievalInferenceConfig создан: n_ctx={retrieval_inference_config.n_ctx}, model_path={retrieval_inference_config.model_path.name}")

    # НОВОЕ: Загрузка конфигурации для запасного инференса ретривера
    retrieval_fallback_inference_conf_data = data.get("retrieval_fallback_inference", None)
    retrieval_fallback_inference_config: Optional[InferenceConfig] = None
    if retrieval_fallback_inference_conf_data:
        logger.debug(f"Сырые данные retrieval_fallback_inference: {retrieval_fallback_inference_conf_data}")
        retrieval_fallback_inference_config = InferenceConfig(
            engine_type=retrieval_fallback_inference_conf_data.get("engine_type", "llamacpp"),
            model_path=Path(retrieval_fallback_inference_conf_data.get("model_path", "models/default_fallback_model.gguf")),
            n_gpu_layers=retrieval_fallback_inference_conf_data.get("n_gpu_layers", 0),
            device_type=retrieval_fallback_inference_conf_data.get("device_type", "auto"),
            n_ctx=retrieval_fallback_inference_conf_data.get("n_ctx", 4096), 
            temperature=retrieval_fallback_inference_conf_data.get("temperature", 0.01), 
            max_new_tokens=retrieval_fallback_inference_conf_data.get("max_new_tokens", 5), 
            top_p=retrieval_fallback_inference_conf_data.get("top_p", 0.1), 
            top_k=retrieval_fallback_inference_conf_data.get("top_k", 1), 
            repeat_penalty=retrieval_fallback_inference_conf_data.get("repeat_penalty", 1.0), 
            stop_sequences=retrieval_fallback_inference_conf_data.get("stop_sequences", ["\n", ".", ",", "!", "?", "Да.", "Нет."]), 
        )
        logger.debug(f"RetrievalFallbackInferenceConfig создан: n_ctx={retrieval_fallback_inference_config.n_ctx}, model_path={retrieval_fallback_inference_config.model_path.name}")


    # Загрузка конфигурации для инференса синтеза
    synthesis_inference_conf_data = data.get("synthesis_inference", {})
    logger.debug(f"Сырые данные synthesis_inference: {synthesis_inference_conf_data}")
    synthesis_inference_config = InferenceConfig(
        engine_type=synthesis_inference_conf_data.get("engine_type", "llamacpp"),
        model_path=Path(synthesis_inference_conf_data.get("model_path", "models/default_synthesis_model.gguf")),
        n_gpu_layers=synthesis_inference_conf_data.get("n_gpu_layers", 0),
        device_type=synthesis_inference_conf_data.get("device_type", "auto"),
        n_ctx=synthesis_inference_conf_data.get("n_ctx", 16384), 
        temperature=synthesis_inference_conf_data.get("temperature", 0.7),
        max_new_tokens=synthesis_inference_conf_data.get("max_new_tokens", 500),
        top_p=synthesis_inference_conf_data.get("top_p", 0.95),
        top_k=synthesis_inference_conf_data.get("top_k", 40),
        repeat_penalty=synthesis_inference_conf_data.get("repeat_penalty", 1.1),
        stop_sequences=synthesis_inference_conf_data.get("stop_sequences", ["\n\nВопрос:", "###", "User:"]),
    )
    logger.debug(f"SynthesisInferenceConfig создан: n_ctx={synthesis_inference_config.n_ctx}, model_path={synthesis_inference_config.model_path.name}")


    return RAGConfig(
        chunking_config=chunking_config,
        retrieval_config=retrieval_config,
        synthesis_config=synthesis_config,
        retrieval_inference_config=retrieval_inference_config,
        synthesis_inference_config=synthesis_inference_config,
        retrieval_fallback_inference_config=retrieval_fallback_inference_config, # НОВОЕ: Передаем конфиг запасного ретривера
        general_language=data.get("general_language", "ru") 
    )

def load_config(config_path: Path) -> Dict[str, Any]:
    """Загружает основную конфигурацию приложения из JSON файла."""
    logger.info(f"Загружаю основную конфигурацию из: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.debug(f"Сырые данные основной конфига из файла: {json.dumps(config, indent=2, ensure_ascii=False)}")
    except FileNotFoundError:
        logger.critical(f"Файл основной конфигурации не найден: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.critical(f"Ошибка декодирования JSON в файле основной конфигурации {config_path}: {e}")
        raise

    # Применяем дефолтные значения, если они отсутствуют
    default_config = {
        "language": "ru", 
        "interface": "cli",
        "logging": {
            "level": "INFO", 
            "log_to_console": True,
            "console_level": "INFO", 
            "log_to_file": True,
            "log_file_path": "logs/app.log"
        }
    }
    
    # Рекурсивное обновление словаря с дефолтными значениями
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = u[k]
        return d
    
    config = update_dict(default_config, config)
    
    logger.debug(f"Загруженная основная конфигурация: {config}")
    return config
