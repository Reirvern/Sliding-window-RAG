import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Any # Импортируем Any
from core.utils.localization.translator import Translator
from core.domain.models import RAGQuery # Импортируем DTO
from core.utils.observer import Observer # Импортируем Observer

class CLIInterface(Observer): # CLIInterface должен быть Observer
    """Интерфейс командной строки для Alt-RAG"""
    
    def __init__(self, config: dict, rag_engine, logger, translator: Translator):
        super().__init__() # Убедимся, что Observer инициализирован
        self.config = config
        self.rag_engine = rag_engine
        self.logger = logger
        self.translator = translator
        self.output_dir = None
        self.progress_bars = {} # Словарь для хранения активных прогресс-баров

    def run(self):
        """Основной цикл CLI интерфейса"""
        # Приветствие
        print(self.translator.translate("welcome")) # Здесь print() допустим, т.к. баров еще нет
        
        # Создание папки для результатов
        self._create_output_folder()
        
        # Запрос пути к файлам
        input_path_str = self._get_input_path()
        input_path = Path(input_path_str) # Преобразуем в Path объект
        
        # Запрос вопроса
        question = self._get_question()
        
        # Сохранение запроса
        self._save_query(question)
        
        # Создаем DTO запроса
        rag_query = RAGQuery(question=question, input_path=input_path, output_dir=self.output_dir)

        # Запуск обработки с прогресс-барами (теперь через RAGEngine)
        self.logger.info(self.translator.translate("processing"))
        tqdm.write("\n" + self.translator.translate("processing")) # Используем tqdm.write()
        
        final_answer = self.rag_engine.run(rag_query) # Запускаем RAG Engine

        # Вывод результата
        tqdm.write("\n" + self.translator.translate("result_title"))
        tqdm.write("-" * 50)
        tqdm.write(final_answer)
        tqdm.write("-" * 50)
        
    def _create_output_folder(self):
        """Создает уникальную папку для результатов"""
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        base_dir = Path("user_data") / "outputs"
        self.output_dir = base_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(self.translator.translate("folder_created").format(path=self.output_dir))
        print(self.translator.translate("folder_created").format(path=self.output_dir)) # Здесь print() допустим
        
    def _get_input_path(self) -> str:
        """Запрашивает у пользователя путь к файлам"""
        default_path = "input/"
        input_path_str = input(self.translator.translate("input_path_prompt"))
        if not input_path_str:
            input_path_str = default_path
        
        # Проверка существования пути
        if not Path(input_path_str).exists():
            self.logger.warning(self.translator.translate("invalid_path").format(path=input_path_str))
            print(self.translator.translate("invalid_path")) # Здесь print() допустим
            input_path_str = default_path # Возвращаемся к дефолтному, если путь невалиден
            
        return input_path_str
        
    def _get_question(self) -> str:
        """Запрашивает у пользователя вопрос"""
        return input(self.translator.translate("question_prompt"))
        
    def _save_query(self, question: str):
        """Сохраняет запрос пользователя в файл"""
        query_dir = self.output_dir / "query"
        query_dir.mkdir(parents=True, exist_ok=True)
        query_file_path = query_dir / "query.json"
        
        query_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question
        }
        
        try:
            with open(query_file_path, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Запрос сохранен в {query_file_path}")
        except Exception as e:
            self.logger.error(self.translator.translate("error_saving_query").format(file=query_file_path), exc_info=True)
            print(self.translator.translate("error_saving_query").format(file=query_file_path))

    def update(self, message_type: str, data: Any):
        """
        Обрабатывает уведомления от наблюдаемых объектов (RAGEngine, сервисов).
        """
        if message_type == "progress":
            stage = data.get("stage")
            
            if stage == "chunking":
                # Получаем данные о прогрессе
                current_file_index = data.get("current_file_index")
                total_files = data.get("total_files")
                file_name = data.get("file_name")
                current_chunk_in_file = data.get("current_chunk_in_file")
                total_chunks_in_file = data.get("total_chunks_in_file")
                overall_progress_percent = data.get("file_progress_percent") # Это общий процент

                # Создаем/получаем общий прогресс-бар для чанкинга (по процентам)
                if "chunking_overall" not in self.progress_bars:
                    self.progress_bars["chunking_overall"] = tqdm(
                        total=100, # Прогресс от 0 до 100%
                        desc=self.translator.translate("progress_chunking_overall"), 
                        unit="%", 
                        leave=True, # Оставляем бар после завершения
                        position=0 # Верхняя позиция
                    )
                
                pbar = self.progress_bars["chunking_overall"]
                # Обновляем прогресс-бар до текущего процента
                pbar.n = overall_progress_percent
                pbar.refresh() # Принудительное обновление для немедленного отображения
                
                # Обновляем описание бара для более детальной информации
                pbar.set_description(
                    self.translator.translate("progress_chunking", current=current_file_index, total=total_files) + 
                    f" ({file_name}, {current_chunk_in_file}/{total_chunks_in_file} chunks)"
                )

            elif stage == "retrieval":
                current = data.get("current")
                total = data.get("total")
                
                if "retrieval_overall" not in self.progress_bars:
                    self.progress_bars["retrieval_overall"] = tqdm(
                        total=total, 
                        desc=self.translator.translate("progress_retrieval_overall"), 
                        unit="chunk", 
                        leave=True,
                        position=0 # Верхняя позиция
                    )
                
                pbar = self.progress_bars["retrieval_overall"]
                if pbar.n < current:
                    pbar.update(current - pbar.n)
                
                percent = int((current / total) * 100)
                pbar.set_description(self.translator.translate("progress_retrieval", percent=percent))

            elif stage == "synthesis":
                current = data.get("current")
                total = data.get("total")
                
                if "synthesis_overall" not in self.progress_bars:
                    self.progress_bars["synthesis_overall"] = tqdm(
                        total=total, 
                        desc=self.translator.translate("progress_synthesis_overall"), 
                        unit="step", 
                        leave=True,
                        position=0 # Верхняя позиция
                    )
                
                pbar = self.progress_bars["synthesis_overall"]
                if pbar.n < current:
                    pbar.update(current - pbar.n)
                
                percent = int((current / total) * 100)
                pbar.set_description(self.translator.translate("progress_processing", percent=percent))
            
        elif message_type == "complete":
            stage = data.get("stage")
            
            # Закрываем соответствующий прогресс-бар, если он существует
            pbar_key = f"{stage}_overall"
            if pbar_key in self.progress_bars:
                self.progress_bars[pbar_key].close()
                del self.progress_bars[pbar_key]
            
            # Выводим сообщение о завершении этапа
            if stage == "chunking":
                tqdm.write(self.translator.translate("chunking_complete_log").format(chunks=data.get("total_chunks")))
            elif stage == "retrieval":
                tqdm.write(self.translator.translate("retrieval_complete_log").format(chunks=data.get("relevant_chunks_count")))
            elif stage == "synthesis":
                tqdm.write(self.translator.translate("synthesis_complete_log"))
            
            # Финальное сообщение о завершении RAG процесса
            if stage == "rag_process": 
                tqdm.write(self.translator.translate("rag_process_complete_log"))
                # Убедимся, что все бары закрыты
                for pbar_key_left in list(self.progress_bars.keys()):
                    self.progress_bars[pbar_key_left].close()
                    del self.progress_bars[pbar_key_left]

        elif message_type == "status":
            message = data.get("message")
            tqdm.write(message)

        elif message_type == "error":
            stage = data.get("stage")
            error_msg = data.get("error")
            self.logger.error(self.translator.translate("error_in_stage").format(stage=stage, error=error_msg))
            tqdm.write(self.translator.translate("error_in_stage").format(stage=stage, error=error_msg))

