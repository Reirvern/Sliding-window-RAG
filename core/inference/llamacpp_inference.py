# core/inference/llamacpp_inference.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import gc 

try:
    from llama_cpp import Llama 
except ImportError:
    logging.getLogger('AltRAG').warning("Библиотека 'llama-cpp-python' не найдена. Убедитесь, что она установлена.")
    Llama = None 

from core.inference.base_inference import BaseInferenceEngine
from core.domain.models import InferenceConfig

class LlamacppInferenceEngine(BaseInferenceEngine):
    """
    Инференс-движок для работы с моделями GGUF через llama-cpp-python.
    """
    def __init__(self, config: InferenceConfig, logger: logging.Logger):
        super().__init__(config, logger)
        if Llama is None:
            raise RuntimeError("llama-cpp-python не установлен или не может быть импортирован. Пожалуйста, установите его (например, 'pip install llama-cpp-python[cuda]').")
        self._loaded_model: Optional[Llama] = None

    def load_model(self) -> Llama:
        """
        Загружает модель GGUF в память с учетом n_gpu_layers, device_type и n_ctx.
        Добавляет chat_format для правильного форматирования промптов.
        """
        if self._loaded_model:
            self.logger.info(f"Модель {self.config.model_path.name} уже загружена.")
            return self._loaded_model

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {self.config.model_path}")

        self.logger.info(f"Загружаю модель {self.config.model_path.name} с {self.config.n_gpu_layers} слоями на GPU ({self.config.device_type})...")
        
        self.logger.debug(f"Конфигурация для загрузки модели: n_ctx={self.config.n_ctx}, n_gpu_layers={self.config.n_gpu_layers}")

        # Параметры для Llama
        llama_params = {
            "model_path": str(self.config.model_path),
            "n_gpu_layers": self.config.n_gpu_layers,
            "n_ctx": self.config.n_ctx,
            "verbose": False, 
            "chat_format": "gemma", # Явно указываем chat_format для моделей Gemma
        }
        
        try:
            self._loaded_model = Llama(**llama_params)
            self.logger.info(f"Модель {self.config.model_path.name} успешно загружена.")
            
            if hasattr(self._loaded_model, 'n_ctx_train') and self._loaded_model.n_ctx() < self._loaded_model.n_ctx_train():
                self.logger.warning(
                    f"Контекстное окно модели ({self._loaded_model.n_ctx()}) меньше, чем контекст, на котором модель была обучена ({self._loaded_model.n_ctx_train()}). "
                    "Это не ошибка, но может означать, что вы не используете весь потенциал модели."
                )
            return self._loaded_model
        except Exception as e:
            self.logger.critical(f"Ошибка при загрузке модели {self.config.model_path.name}: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось загрузить модель: {e}")

    def generate(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        """
        Генерирует текст, используя загруженную модель, с учетом чат-формата.
        :param messages: Список сообщений в формате [{'role': 'user', 'content': '...'}]
        :param gen_kwargs: Дополнительные параметры генерации (temperature, max_tokens и т.д.)
        """
        if not self._loaded_model:
            self.logger.error(f"Модель {self.config.model_path.name} не загружена перед вызовом generate. Это ошибка в логике.")
            return "Ошибка: Модель не загружена."

        gen_params = self._apply_generation_params(gen_kwargs)

        # Вывод полного промпта, сгенерированного llama-cpp-python
        # Мы не формируем его сами, это делает create_chat_completion
        # Для отладки, можем логировать входные сообщения
        self.logger.debug(f"Входные сообщения для LLM (create_chat_completion): {messages}")
        self.logger.debug(f"Параметры генерации: {gen_params}")

        try:
            output = self._loaded_model.create_chat_completion(
                messages=messages,
                temperature=gen_params.get("temperature"),
                max_tokens=gen_params.get("max_tokens"),
                top_p=gen_params.get("top_p"),
                top_k=gen_params.get("top_k"),
                repeat_penalty=gen_params.get("repeat_penalty"),
                stop=gen_params.get("stop"),
            )
            
            self.logger.debug(f"Полный ответ от create_chat_completion: {output}")
            if "choices" in output and len(output["choices"]) > 0:
                generated_text = output["choices"][0].get("message", {}).get("content", "")
                self.logger.debug(f"Извлеченный сгенерированный текст: {generated_text.strip()[:200]}...")
                if not generated_text:
                    self.logger.warning("Сгенерированный текст пуст!")
            else:
                generated_text = ""
                self.logger.warning("Ответ 'choices' от LLM пуст или отсутствует.")

            return generated_text
        except Exception as e:
            self.logger.error(f"Ошибка при генерации текста: {e}", exc_info=True)
            return f"Ошибка генерации: {e}"

    def unload_model(self):
        """
        Выгружает модель из памяти.
        """
        if self._loaded_model:
            self.logger.info(f"Выгружаю модель {self.config.model_path.name} из памяти.")
            del self._loaded_model
            self._loaded_model = None
            gc.collect() 
        else:
            self.logger.info("Модель не загружена, выгрузка не требуется.")

