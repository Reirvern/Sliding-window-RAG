import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from core.utils.localization.translator import Translator
from core.domain.models import RAGQuery # Импортируем DTO
from core.utils.observer import Observer # Импортируем Observer

class CLIInterface(Observer): # CLIInterface теперь является наблюдателем
    """Интерфейс командной строки для Alt-RAG"""
    
    def __init__(self, config: dict, rag_engine, logger, translator: Translator):
        super().__init__() # Вызываем инициализатор Observer
        self.config = config
        self.rag_engine = rag_engine
        self.logger = logger
        self.translator = translator
        self.output_dir = None
        self.progress_bars = {} # Для управления прогресс-барами tqdm
        
        # Регистрируем CLIInterface как наблюдателя RAGEngine
        self.rag_engine.add_observer(self)

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
        
        # Создание объекта RAGQuery DTO
        rag_query = RAGQuery(
            question=question,
            input_path=input_path,
            output_dir=self.output_dir
        )
        
        # Сохранение запроса через DTO
        self._save_query(rag_query)
        
        # Запуск обработки
        print("\n" + self.translator.translate("processing"))
        
        # Запускаем RAGEngine и ждем ответа
        final_answer = self.rag_engine.run(rag_query) # Вызов RAGEngine
        
        # Вывод результата
        self._display_final_result(final_answer)
        
    def _create_output_folder(self):
        """Создает уникальную папку для результатов"""
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        base_dir = Path("user_data") / "outputs" / timestamp
        self.output_dir = base_dir
        
        # Создаем подпапки
        (base_dir / "query").mkdir(parents=True, exist_ok=True)
        (base_dir / "chunks").mkdir(exist_ok=True)
        (base_dir / "relevant_chunks").mkdir(exist_ok=True)
        (base_dir / "analysis").mkdir(exist_ok=True)
        (base_dir / "answer").mkdir(exist_ok=True)
        
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
            print(f"Используется путь по умолчанию: {default_path}") 
            return default_path
        
        input_path = Path(user_input)
        if not input_path.exists():
            print(self.translator.translate("invalid_path"))
            print(f"Будет использован путь по умолчанию: {default_path}") 
            return default_path
        
        return input_path
    
    def _get_question(self) -> str:
        """Запрашивает вопрос у пользователя"""
        prompt = self.translator.translate("question_prompt")
        return input(prompt).strip()
    
    def _save_query(self, rag_query: RAGQuery):
        """Сохраняет запрос в папку query"""
        query_data = {
            "timestamp": datetime.now().isoformat(),
            "question": rag_query.question, 
            "input_path": str(rag_query.input_path) 
        }
        
        query_file = rag_query.output_dir / "query" / "query.json" 
        try:
            with open(query_file, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Запрос сохранен в {query_file}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении запроса в {query_file}: {e}")
            print(self.translator.translate("error_saving_query").format(file=query_file))

    def _display_final_result(self, result: str):
        """Выводит финальный результат пользователю."""
        print("\n" + self.translator.translate("result_title"))
        print("-" * 50)
        print(result)
        print("-" * 50)

    # interface/cli.py (фрагмент в методе update)
    # ... (остальной импорт и инициализация) ...

    def update(self, message_type: str, data: any):
        """
        Обрабатывает уведомления от наблюдаемых объектов (RAGEngine, сервисов).
        """
        if message_type == "progress":
            stage = data.get("stage")
            
            if stage == "chunking":
                current_file = data.get("current")
                total_files = data.get("total")
                file_name = data.get("file_name")
                file_progress_percent = data.get("file_progress_percent") # Новый параметр
                
                # Если прогресс-бара для чанкинга еще нет, создаем его
                if stage not in self.progress_bars:
                    self.progress_bars[stage] = tqdm(total=total_files, desc=self.translator.translate("progress_chunking_overall"), unit="file")
                
                pbar = self.progress_bars[stage]
                # Для чанкинга, мы обновляем бар по файлам, но можем использовать file_progress_percent для более точной desc
                
                # Если это новое "обновление" файла (не просто прогресс внутри файла), то обновляем tqdm.
                # Т.к. `current` инкрементируется за каждый обработанный файл, то `pbar.n` должен соответствовать `current_file - 1`
                if pbar.n < current_file:
                    pbar.update(current_file - pbar.n) # Обновляем на разницу, чтобы не пропускать шаги

                # Обновляем описание прогресс-бара
                pbar.set_description(
                    self.translator.translate("progress_chunking", current=current_file, total=total_files) + 
                    f" ({file_name}, {file_progress_percent}%)" # Добавил имя файла и процент
                )

            elif stage == "retrieval":
                current = data.get("current")
                total = data.get("total")
                if stage not in self.progress_bars:
                    self.progress_bars[stage] = tqdm(total=total, desc=self.translator.translate("progress_retrieval_overall"), unit="chunk")
                
                pbar = self.progress_bars[stage]
                pbar.update(1)
                percent = int((current / total) * 100)
                pbar.set_description(self.translator.translate("progress_retrieval", percent=percent))

            elif stage == "synthesis":
                current = data.get("current")
                total = data.get("total")
                if stage not in self.progress_bars:
                    self.progress_bars[stage] = tqdm(total=total, desc=self.translator.translate("progress_synthesis_overall"), unit="step")
                
                pbar = self.progress_bars[stage]
                pbar.update(1)
                percent = int((current / total) * 100)
                pbar.set_description(self.translator.translate("progress_processing", percent=percent))
            
            # ... (остальные типы сообщений: complete, status, error) ...
            
        elif message_type == "complete":
            stage = data.get("stage")
            if stage in self.progress_bars:
                self.progress_bars[stage].close() # Закрываем прогресс-бар
                del self.progress_bars[stage]
            
            # Выводим сообщение о завершении этапа
            if stage == "chunking":
                self.logger.info(self.translator.translate("chunking_complete_log").format(chunks=data.get("total_chunks")))
            elif stage == "retrieval":
                self.logger.info(self.translator.translate("retrieval_complete_log").format(chunks=data.get("relevant_chunks_count")))
            elif stage == "synthesis":
                self.logger.info(self.translator.translate("synthesis_complete_log"))
            elif stage == "rag_process": 
                self.logger.info(self.translator.translate("rag_process_complete_log"))
                for pbar_key in list(self.progress_bars.keys()):
                    self.progress_bars[pbar_key].close()
                    del self.progress_bars[pbar_key]

        elif message_type == "status":
            message = data.get("message")
            self.logger.info(message)
            print(f"\n{message}")

        elif message_type == "error":
            stage = data.get("stage")
            error_msg = data.get("error")
            self.logger.error(self.translator.translate("error_in_stage").format(stage=stage, error=error_msg))
            print(self.translator.translate("error_in_stage").format(stage=stage, error=error_msg))
