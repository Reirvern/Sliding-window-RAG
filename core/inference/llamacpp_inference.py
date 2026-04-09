# core/inference/llamacpp_inference.py
import logging
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

from core.inference.base_inference import BaseInferenceEngine
from core.domain.models import InferenceConfig

# Импортируем наш детектор железа
from core.utils.hardware_detector import detect_best_runtime

class LlamacppInferenceEngine(BaseInferenceEngine):
    """
    Инференс-движок, который работает через готовый исполняемый файл llama-server.exe
    с поддержкой различных рантаймов (CUDA, Vulkan, CPU).
    """
    def __init__(self, config: InferenceConfig, logger: logging.Logger):
        super().__init__(config, logger)
        self.server_process = None
        self.api_url = "http://127.0.0.1:8080"
        
        project_root = Path(__file__).resolve().parent.parent.parent
        runtimes_dir = project_root / "runtimes"
        
        selected_runtime = getattr(self.config, 'runtime', 'auto')
        
        if selected_runtime == "auto":
            selected_runtime = detect_best_runtime()
            self.logger.info(f"Автовыбор рантайма: {selected_runtime}")
            
        self.llama_server_path = runtimes_dir / selected_runtime / "llama-server.exe"

        if not self.llama_server_path.exists():
            self.logger.warning(f"Исполняемый файл не найден: {self.llama_server_path}")
            self.logger.warning("Откат на базовый рантайм CPU (cpu_x64)...")
            self.llama_server_path = runtimes_dir / "cpu_x64" / "llama-server.exe"
            
            if not self.llama_server_path.exists():
                self.logger.error(f"Критическая ошибка: Базовый рантайм CPU также не найден! Проверьте папку runtimes.")

    def load_model(self) -> bool:
        if self.server_process is not None:
            self.logger.info("Сервер llama.cpp уже запущен.")
            return True

        if not self.llama_server_path.exists():
            raise FileNotFoundError(f"Не найден llama-server.exe по пути: {self.llama_server_path}")

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.config.model_path}")

        self.logger.info(f"Запускаю сервер llama.cpp для модели {self.config.model_path.name}...")

        command = [
            str(self.llama_server_path),
            "-m", str(self.config.model_path),       
            "-c", str(self.config.n_ctx),            
            "-ngl", str(self.config.n_gpu_layers),   
            "--port", "8080",                        
            "--host", "127.0.0.1",                   
            "--reasoning", "off"                     
        ]

        try:
            self.server_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            self.logger.info("Ждем инициализации модели (это может занять несколько секунд)...")
            time.sleep(2) 
            
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.api_url}/health")
                    if response.status_code == 200:
                        self.logger.info("Сервер llama.cpp успешно запущен и готов к работе!")
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)
            
            raise RuntimeError("Сервер llama.cpp не ответил после долгого ожидания.")

        except Exception as e:
            self.logger.critical(f"Ошибка при запуске llama-server: {e}", exc_info=True)
            self.unload_model()
            raise RuntimeError(f"Не удалось запустить сервер: {e}")

    def generate(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        if self.server_process is None:
            self.logger.error("Сервер llama.cpp не запущен перед генерацией. Ошибка в логике.")
            return "Ошибка: Модель не загружена."

        gen_params = self._apply_generation_params(gen_kwargs)

        # Флаг для определения Синтеза (по наличию системного промпта)
        is_complex_chat = any(m.get("role") == "system" for m in messages)

        payload = {
            "temperature": gen_params.get("temperature"),
            "max_tokens": gen_params.get("max_tokens"),
            "top_p": gen_params.get("top_p"),
            "top_k": gen_params.get("top_k"),
            "repeat_penalty": gen_params.get("repeat_penalty"),
            "stop": gen_params.get("stop", []),
            "cache_prompt": False
        }

        try:
            if is_complex_chat:
                self.logger.debug("Определен сложный запрос (Синтез). Использую /v1/chat/completions")
                
                # ИСПРАВЛЕНИЕ ДЛЯ GEMMA: Сливаем роль system и user в одно сообщение
                merged_messages = []
                system_content = ""
                for msg in messages:
                    if msg.get("role") == "system":
                        system_content += msg.get("content", "") + "\n\n"
                    elif msg.get("role") == "user":
                        merged_messages.append({
                            "role": "user", 
                            "content": system_content + msg.get("content", "")
                        })
                        system_content = "" # Очищаем после добавления
                    else:
                        merged_messages.append(msg)
                
                payload["messages"] = merged_messages
                response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload)
            
            else:
                self.logger.debug("Определен простой запрос (Ретривер). Использую /v1/completions")
                raw_prompt = "\n".join([msg.get("content", "") for msg in messages])
                payload["prompt"] = raw_prompt.strip()
                response = requests.post(f"{self.api_url}/v1/completions", json=payload)
            
            # Логируем начало ответа
            self.logger.info("=" * 40)
            self.logger.info(f"СЫРОЙ ОТВЕТ СЕРВЕРА (HTTP {response.status_code}):\n{response.text[:1000]}") 
            self.logger.info("=" * 40)

            response.raise_for_status() 
            result_json = response.json()
            
            if "choices" in result_json and len(result_json["choices"]) > 0:
                if is_complex_chat:
                    generated_text = result_json["choices"][0]["message"]["content"]
                else:
                    generated_text = result_json["choices"][0]["text"]
                return generated_text
            else:
                self.logger.warning("Пустой ответ от сервера 'choices'.")
                return ""
                
        except Exception as e:
            self.logger.error(f"Ошибка при обращении к серверу llama.cpp: {e}", exc_info=True)
            return f"Ошибка генерации: {e}"

    def get_token_count(self, text: str) -> int:
        if self.server_process is None:
            return 0
        try:
            response = requests.post(f"{self.api_url}/tokenize", json={"content": text})
            if response.status_code == 200:
                return len(response.json().get("tokens", []))
            return 0
        except Exception as e:
            self.logger.error(f"Ошибка при подсчете токенов: {e}")
            return 0

    def unload_model(self):
        if self.server_process:
            self.logger.info("Выгружаю модель (останавливаю сервер)...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill() 
            self.server_process = None
            self.logger.info("Модель и сервер выгружены из памяти.")