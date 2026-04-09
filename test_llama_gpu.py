import os
import sys
import gc
from pathlib import Path
import logging

# Настройка базового логгера для вывода в консоль
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LlamaGPOTest')

try:
    from llama_cpp import Llama
    logger.info("Библиотека 'llama-cpp-python' успешно импортирована.")
except ImportError:
    logger.error("Библиотека 'llama-cpp-python' не найдена. Убедитесь, что она установлена в вашем виртуальном окружении.")
    sys.exit(1)

# Укажи путь к твоей модели GGUF
# Убедись, что этот путь верен относительно места запуска скрипта
MODEL_PATH = Path("models/mlabonne_gemma-3-4b-it-abliterated-Q8_0.gguf")

if not MODEL_PATH.exists():
    logger.error(f"Файл модели не найден по пути: {MODEL_PATH.resolve()}")
    sys.exit(1)

# Параметры для загрузки модели с GPU
# Убедись, что n_gpu_layers соответствует твоей видеокарте и модели
N_GPU_LAYERS = 30 # Количество слоев, которые ты хочешь выгрузить на GPU
N_CTX = 6096 # Размер контекстного окна

logger.info(f"Попытка загрузки модели: {MODEL_PATH.name}")
logger.info(f"Слоев на GPU (n_gpu_layers): {N_GPU_LAYERS}")
logger.info(f"Контекст (n_ctx): {N_CTX}")

try:
    # Инициализация модели Llama с явным указанием device="cuda" и verbose=True
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=True, # Включаем подробный вывод
        chat_format="gemma", # Укажи правильный формат чата для твоей модели
        device="cuda", # Явно указываем использовать CUDA
        main_gpu=0, # Явно указываем использовать GPU с индексом 0
    )
    logger.info(f"Модель {MODEL_PATH.name} успешно загружена.")

    # Проверка, сколько слоев было выгружено на GPU
    # Эта информация обычно выводится в консоль самой llama.cpp при verbose=True
    # Но мы также можем попытаться найти ее в логах
    logger.info("Ищите в консоли строку 'llm_load_print_result: ... offloaded X/Y layers to GPU'")

    # Пример генерации текста
    prompt = "Напиши короткий рассказ о коте, который умеет летать."
    messages = [{"role": "user", "content": prompt}]
    
    logger.info(f"Начинаю генерацию для промпта: '{prompt[:50]}...'")
    output = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )
    generated_text = output["choices"][0]["message"]["content"]
    logger.info("Сгенерированный текст:")
    print(generated_text)

except Exception as e:
    logger.critical(f"Произошла ошибка: {e}", exc_info=True)
finally:
    if 'llm' in locals() and llm:
        logger.info("Выгружаю модель из памяти.")
        del llm
        gc.collect()
    logger.info("Тест завершен.")

