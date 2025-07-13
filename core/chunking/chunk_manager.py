import os
import json
from datetime import datetime
from pathlib import Path

class ChunkManager:
    def __init__(self, base_path="user_data", chunk_size=2000):
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.output_dir = self._create_output_dir()
        
    def _create_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        output_dir = self.base_path / timestamp / "chunks"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_chunks(self, chunks: list[dict]):
        """Сохраняет чанки как отдельные JSON-файлы"""
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": f"chunk_{i:04d}",
                "text": chunk["text"],
                "source": chunk["source"],
                "char_count": len(chunk["text"]),
                "created_at": datetime.now().isoformat()
            }
            
            file_path = self.output_dir / f"{chunk_data['id']}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                
        self._save_metadata(len(chunks))
    
    def _save_metadata(self, total_chunks: int):
        meta = {
            "created_at": datetime.now().isoformat(),
            "chunk_size": self.chunk_size,
            "total_chunks": total_chunks,
            "chunk_dir": str(self.output_dir)
        }
        
        with open(self.output_dir.parent / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)