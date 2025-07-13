import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from core.utils.localization.translator import Translator

class CLIInterface:
    """Интерфейс командной строки для Alt-RAG"""
    
    def __init__(self, config: dict, rag_engine, logger, translator: Translator):
        self.config = config
        self.rag_engine = rag_engine
        self.logger = logger
        self.translator = translator
        self.output_dir = None
        
    def run(self):
        """Основной цикл CLI интерфейса"""
        # Приветствие
        print(self.translator.translate("welcome"))
        
        # Создание папки для результатов
        self._create_output_folder()
        
        # Запрос пути к файлам
        input_path = self._get_input_path()
        
        # Запрос вопроса
        question = self._get_question()
        
        # Сохранение запроса
        self._save_query(question)
        
        # Запуск обработки с прогресс-барами
        self._run_with_progress(input_path, question)
        
    def _create_output_folder(self):
        """Создает уникальную папку для результатов"""
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        self.output_dir = Path("user_data") / "outputs" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        msg = self.translator.translate(
            "folder_created", 
            path=str(self.output_dir)
        )
        print(msg)
        self.logger.info(msg)
    
    def _get_input_path(self) -> Path:
        """Запрашивает путь к файлам у пользователя"""
        default_path = Path("input")
        prompt = self.translator.translate("input_path_prompt")
        
        user_input = input(prompt).strip()
        if not user_input:
            return default_path
        
        input_path = Path(user_input)
        if not input_path.exists():
            print(self.translator.translate("invalid_path"))
            return default_path
        
        return input_path
    
    def _get_question(self) -> str:
        """Запрашивает вопрос у пользователя"""
        prompt = self.translator.translate("question_prompt")
        return input(prompt).strip()
    
    def _save_query(self, question: str):
        """Сохраняет запрос в JSON файл"""
        query_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question
        }
        
        query_file = self.output_dir / "query.json"
        with open(query_file, 'w', encoding='utf-8') as f:
            json.dump(query_data, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"Запрос сохранен в {query_file}")
    
    def _run_with_progress(self, input_path: Path, question: str):
        """Запускает обработку с отображением прогресса"""
        print("\n" + self.translator.translate("processing"))
        
        # Имитация обработки с прогресс-барами
        self._simulate_chunking()
        self._simulate_retrieval()
        result = self._simulate_processing()
        
        # Вывод результата
        print("\n" + self.translator.translate("result_title"))
        print("-" * 50)
        print(result)
        print("-" * 50)
    
    def _simulate_chunking(self):
        """Имитация процесса чанкинга с прогресс-баром"""
        total_files = 10
        with tqdm(total=total_files, desc="Чанкирование") as pbar:
            for i in range(total_files):
                time.sleep(0.2)
                pbar.update(1)
                pbar.set_description(
                    self.translator.translate(
                        "progress_chunking",
                        current=i+1,
                        total=total_files
                    )
                )
    
    def _simulate_retrieval(self):
        """Имитация процесса ретривела с прогресс-баром"""
        total_chunks = 100
        with tqdm(total=total_chunks, desc="Поиск фрагментов") as pbar:
            for i in range(total_chunks):
                time.sleep(0.05)
                percent = int((i+1) / total_chunks * 100)
                pbar.update(1)
                pbar.set_description(
                    self.translator.translate(
                        "progress_retrieval",
                        percent=percent
                    )
                )
                
                # Расчет и отображение оставшегося времени
                if i > 10:  # После первых 10 итераций
                    elapsed = time.time() - pbar.start_t
                    time_per_item = elapsed / (i+1)
                    remaining = time_per_item * (total_chunks - i - 1)
                    mins, secs = divmod(int(remaining), 60)
                    
                    pbar.set_postfix_str(
                        self.translator.translate(
                            "time_remaining",
                            minutes=mins,
                            seconds=secs
                        )
                    )
    
    def _simulate_processing(self) -> str:
        """Имитация финальной обработки с прогресс-баром"""
        total_steps = 50
        result = ""
        
        with tqdm(total=total_steps, desc="Формирование ответа") as pbar:
            for i in range(total_steps):
                time.sleep(0.1)
                percent = int((i+1) / total_steps * 100)
                pbar.update(1)
                pbar.set_description(
                    self.translator.translate(
                        "progress_processing",
                        percent=percent
                    )
                )
                
                # Постепенно "формируем" ответ
                if i < 10:
                    result += "Формирование ответа... "
                elif i < 40:
                    result += "Анализ релевантных фрагментов... "
        
        return result + "Итоговый ответ на ваш вопрос."