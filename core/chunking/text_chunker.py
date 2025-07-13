import json
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
from pathlib import Path
import logging

class TextChunker:
    def __init__(self, chunk_size=2000):
        self.chunk_size = chunk_size
        self.setup_nltk()
    
    def setup_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab/russian')
        except LookupError:
            print("Загрузка ресурсов NLTK для русского языка...")
            nltk.download('punkt', quiet=False)
            nltk.download('punkt_tab', quiet=False)
    
    def chunk_file(self, file_path: Path, output_dir: Path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sentences = sent_tokenize(text, language='russian')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sent_length = len(sentence)
            
            # Обработка слишком длинных предложений
            if sent_length > self.chunk_size:
                print(f"Обнаружено длинное предложение ({sent_length} символов)")
                for i in range(0, sent_length, self.chunk_size):
                    part = sentence[i:i+self.chunk_size]
                    chunks.append(part)
                continue
            
            if current_length + sent_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sent_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = file_path.stem
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk,
                "source_file": str(file_path),
                "chunk_id": f"{base_name}_{i+1}",
                "created_at": datetime.now().isoformat(),
                "token_count": len(chunk.split())
            }
            
            chunk_file = output_dir / f"{base_name}_chunk_{i+1:04d}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        return len(chunks)

def process_text_files():
    # Определяем пути относительно расположения скрипта
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Создаем папку input, если ее нет
    input_dir = project_root / "input"
    input_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%d.%m.%y.%H.%M.%S")
    output_base = project_root / "user_data" / timestamp / "chunks"
    
    total_chunks = 0
    processed_files = 0
    
    print(f"Поиск файлов в: {input_dir}")
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"Не найдено .txt файлов в папке {input_dir}")
        print("Поместите текстовые файлы в эту папку и запустите скрипт снова.")
        return
    
    for file_path in txt_files:
        print(f"Обработка файла: {file_path.name}")
        num_chunks = TextChunker().chunk_file(file_path, output_base)
        total_chunks += num_chunks
        processed_files += 1
        print(f"  Создано чанков: {num_chunks}")
    
    print(f"\nИтоги обработки:")
    print(f"  Обработано файлов: {processed_files}")
    print(f"  Всего создано чанков: {total_chunks}")
    print(f"  Результаты сохранены в: {output_base}")

if __name__ == "__main__":
    print("Запуск текст-чанкера...")
    process_text_files()
    print("Обработка завершена.")