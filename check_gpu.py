# check_gpu.py
import os
import sys
import logging
from pathlib import Path

# Настройка базового логгера, чтобы видеть вывод llama.cpp
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    logger.info("Библиотека 'llama_cpp' успешно импортирована.")
    # НОВОЕ: Выводим установленную версию llama-cpp-python
    if hasattr(Llama, '__version__'):
        logger.info(f"Установленная версия llama-cpp-python: {Llama.__version__}")
    else:
        logger.warning("Не удалось определить версию llama-cpp-python (отсутствует атрибут __version__).")
except ImportError:
    logger.error("Ошибка: Библиотека 'llama_cpp' не найдена. Убедитесь, что она установлена.")
    sys.exit(1)

# Укажи здесь правильный путь к твоей GGUF модели
# Убедись, что файл модели существует по этому пути
# Например: Path("models") / "mlabonne_gemma-3-4b-it-abliterated-Q8_0.gguf"
# или Path("C:/path/to/your/models/your_model.gguf")
MODEL_RELATIVE_PATH = Path("models") / "mlabonne_gemma-3-4b-it-abliterated-Q8_0.gguf"
# Определяем корневую директорию проекта (где лежит main.py и папка models)
PROJECT_ROOT = Path(__file__).parent
model_path = PROJECT_ROOT / MODEL_RELATIVE_PATH

logger.info(f"Попытка загрузить модель: {model_path}")

if not model_path.exists():
    logger.error(f"Ошибка: Файл модели не найден по пути: {model_path}")
    logger.info("Пожалуйста, убедитесь, что путь к модели в 'check_gpu.py' указан верно.")
    sys.exit(1)

try:
    # Попытка загрузить модель с выгрузкой всех слоев на GPU (-1)
    # verbose=True включит подробный вывод от llama.cpp
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=-1, # -1 означает попытку выгрузить все слои на GPU
        verbose=True     # Включает подробный вывод от llama.cpp
    )
    
    logger.info(f"Модель успешно загружена: {model_path.name}")
    
    # Проверка, сколько слоев фактически загружено на GPU
    # В новых версиях llama-cpp-python есть атрибут n_gpu_layers
    # В старых версиях его может не быть, но verbose=True покажет информацию
    if hasattr(llm, 'n_gpu_layers'):
        actual_gpu_layers = llm.n_gpu_layers
        logger.info(f"Количество слоев, загруженных на GPU: {actual_gpu_layers}")
        if actual_gpu_layers > 0:
            logger.info("GPU-ускорение, вероятно, работает!")
        else:
            logger.warning("Слои не были загружены на GPU. Проверьте настройки n_gpu_layers и наличие CUDA.")
    else:
        logger.info("Не удалось получить количество слоев на GPU через атрибут n_gpu_layers (возможно, старая версия llama_cpp_python).")
        logger.info("Пожалуйста, внимательно изучите лог выше на предмет упоминаний 'CUDA devices' или 'ggml_cuda_init'.")

    # Выгружаем модель из памяти
    # НОВОЕ: Добавлена проверка наличия unload_model()
    if hasattr(llm, 'unload_model'):
        llm.unload_model()
        logger.info("Модель выгружена из памяти.")
    else:
        logger.warning("Метод 'unload_model()' не найден (возможно, старая версия llama_cpp_python). Пропускаю выгрузку.")
    
except Exception as e:
    logger.critical(f"Критическая ошибка при загрузке или работе модели: {e}", exc_info=True)
    logger.error("GPU-ускорение, возможно, не работает или возникла другая проблема.")
    sys.exit(1)
