# core/services/synthesis_service.py
import logging
from typing import List, Any, Dict # Добавляем Dict
from core.domain.models import RAGQuery, Chunk, SynthesisConfig, InferenceConfig
from core.utils.observer import Observable, Observer
from core.utils.localization.translator import Translator

class SynthesisService(Observable, Observer):
    """
    Сервис для синтеза финального ответа на основе релевантных чанков.
    """
    def __init__(self, 
                 config: SynthesisConfig, 
                 logger: logging.Logger, 
                 translator: Translator,
                 inference_engine):
        super().__init__()
        self.config = config
        self.logger = logger
        self.translator = translator
        self.inference_engine = inference_engine # Инференс-движок для синтеза

    def synthesize_answer(self, rag_query: RAGQuery, relevant_chunks: List[Chunk]) -> str:
        """
        Синтезирует финальный ответ, используя релевантные чанки и LLM.
        """
        self.logger.info("Начинаю синтез ответа.")
        self.notify_observers("status", {"message": self.translator.translate("synthesis_in_progress")})

        # Объединяем содержимое релевантных чанков в один контекст
        context = "\n\n".join([chunk.content for chunk in relevant_chunks])
        
        try:
            formatted_user_content = self.config.prompt_template.format(
                context=context,
                question=rag_query.question
            )
        except KeyError as e:
            self.logger.error(f"Ошибка форматирования промпта для синтеза: отсутствует ключ {e}. Проверьте prompt_template в конфиге SynthesisConfig.")
            return f"Ошибка форматирования промпта для синтеза: отсутствует ключ {e}."

        # Формируем список сообщений для create_chat_completion
        messages: List[Dict[str, str]] = [ # Явно указываем тип
            {"role": "user", "content": formatted_user_content}
        ]
        
        # --- ДОБАВЛЕНО ДЛЯ ОТЛАДКИ ---
        self.logger.debug(f"Type of 'messages' list before LLM call: {type(messages)}")
        for i, msg in enumerate(messages):
            self.logger.debug(f"Message {i} type: {type(msg)}")
            if not isinstance(msg, dict):
                self.logger.error(f"ERROR: Message at index {i} is NOT a dictionary. Type: {type(msg)}, Value: {msg}")
                # Дополнительная проверка, которая должна поймать проблему
                raise TypeError(f"Message at index {i} in 'messages' list is not a dictionary (it's {type(msg)}). Expected Dict.")
            
            self.logger.debug(f"Message {i} keys: {msg.keys()}")
            if "role" not in msg:
                self.logger.error(f"ERROR: Message at index {i} is missing 'role' key. Value: {msg}")
                raise ValueError(f"Message at index {i} in 'messages' list is missing 'role' key.")
            if "content" not in msg:
                self.logger.error(f"ERROR: Message at index {i} is missing 'content' key. Value: {msg}")
                raise ValueError(f"Message at index {i} in 'messages' list is missing 'content' key.")
            
            self.logger.debug(f"Message {i} role: {msg.get('role')}")
            self.logger.debug(f"Message {i} content type: {type(msg.get('content'))}")
            self.logger.debug(f"Message {i} content (first 200 chars): {str(msg.get('content'))[:200]}...")
        # --- КОНЕЦ ДОБАВЛЕНИЯ ДЛЯ ОТЛАДКИ ---

        self.logger.debug(f"Промпт для генерации (сообщения): {messages}")

        try:
            final_answer = self.inference_engine.generate(
                messages=messages,
                temperature=self.inference_engine.config.temperature,
                max_new_tokens=self.inference_engine.config.max_new_tokens,
                top_p=self.inference_engine.config.top_p,
                top_k=self.inference_engine.config.top_k,
                repeat_penalty=self.inference_engine.config.repeat_penalty,
                stop=self.inference_engine.config.stop_sequences
            )
            self.logger.info("Синтез ответа завершен.")
            return final_answer
        except Exception as e:
            self.logger.error(f"Ошибка при генерации финального ответа: {e}", exc_info=True)
            self.notify_observers("error", {"stage": "synthesis", "error": str(e)})
            return self.translator.translate("synthesis_error").format(error=str(e))

    def update(self, message_type: str, data: Any):
        self.notify_observers(message_type, data)
        if message_type != "progress":
            self.logger.debug(f"SynthesisService получил уведомление: Type={message_type}, Data={data}")
