import json
from pathlib import Path
from core.domain.models import RAGConfig, ChunkingConfig, RetrievalConfig, SynthesisConfig

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_rag_config(config_path: Path) -> RAGConfig:
    """Загружает конфигурацию RAG из JSON файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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

    retrieval_conf_data = data.get("retrieval", {})
    retrieval_config = RetrievalConfig(
        strategy_type=retrieval_conf_data.get("strategy_type", 1),
        model_path=Path(retrieval_conf_data["model_path"]) if retrieval_conf_data.get("model_path") else None,
        keywords=retrieval_conf_data.get("keywords"),
        top_k=retrieval_conf_data.get("top_k", 5)
    )

    synthesis_conf_data = data.get("synthesis", {})
    synthesis_config = SynthesisConfig(
        model_path=Path(synthesis_conf_data["model_path"]) if synthesis_conf_data.get("model_path") else None,
        temperature=synthesis_conf_data.get("temperature", 0.7),
        max_new_tokens=synthesis_conf_data.get("max_new_tokens", 500),
        prompt_template=synthesis_conf_data.get("prompt_template", "")
    )

    return RAGConfig(
        chunking_config=chunking_config,
        retrieval_config=retrieval_config,
        synthesis_config=synthesis_config,
        general_language=data.get("general_language", "en")
    )