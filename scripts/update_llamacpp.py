# scripts/update_llamacpp.py
import os
import sys
import json
import urllib.request
import zipfile
import io
import shutil
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LlamaUpdater')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIMES_DIR = PROJECT_ROOT / "runtimes"

GITHUB_API_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

# Обновленные правила поиска нужного архива для каждой папки
TARGET_MAPPING = {
    # Для CPU ищем базовый win-x64.zip, исключая все GPU-специфичные (cuda, vulkan, sycl и т.д.)
    "cpu_x64": lambda n: "win" in n.lower() and "x64" in n.lower() and all(ext not in n.lower() for ext in ["cuda", "vulkan", "sycl", "hip", "rpc", "cudart", "openvino"]),
    
    # Для CUDA 12 ищем упоминание cuda и (12. или cu12). Исключаем cudart (это просто библиотеки среды, а не сам llama.cpp)
    "cuda12": lambda n: "win" in n.lower() and "cuda" in n.lower() and ("12." in n.lower() or "cu12" in n.lower()) and "x64" in n.lower() and "cudart" not in n.lower(),
    
    # Для CUDA 13 ищем упоминание cuda и (13. или cu13).
    "cuda13": lambda n: "win" in n.lower() and "cuda" in n.lower() and ("13." in n.lower() or "cu13" in n.lower()) and "x64" in n.lower() and "cudart" not in n.lower(),
    
    # Для Vulkan правило осталось простым
    "vulkan": lambda n: "win" in n.lower() and "vulkan" in n.lower() and "x64" in n.lower() and "cudart" not in n.lower()
}

def get_latest_release_info():
    """Получает информацию о последнем релизе через GitHub API"""
    logger.info("Запрашиваю информацию о последнем релизе llama.cpp...")
    req = urllib.request.Request(GITHUB_API_URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        logger.error(f"Ошибка при обращении к GitHub API: {e}")
        sys.exit(1)

def download_and_extract(url, target_dir):
    """Скачивает zip-архив в память и извлекает .exe и .dll в нужную папку"""
    logger.info(f"Скачиваю: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
            
        logger.info(f"Распаковка в {target_dir.name}...")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            for file_info in z.infolist():
                # Нас интересуют только исполняемые файлы и библиотеки
                if file_info.filename.endswith('.exe') or file_info.filename.endswith('.dll'):
                    filename = Path(file_info.filename).name
                    target_path = target_dir / filename
                    
                    with z.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
        logger.info(f"Папка {target_dir.name} успешно обновлена.")
    except PermissionError:
        logger.error(f"ОШИБКА ДОСТУПА: Не могу перезаписать файлы в {target_dir.name}. "
                     f"Убедись, что llama-server.exe сейчас не запущен!")
    except Exception as e:
        logger.error(f"Ошибка при скачивании или распаковке {url}: {e}")

def main():
    if not RUNTIMES_DIR.exists():
        logger.error(f"Директория {RUNTIMES_DIR} не найдена. Убедитесь, что скрипт лежит в папке scripts/")
        sys.exit(1)

    release_info = get_latest_release_info()
    tag_name = release_info.get('tag_name', 'Unknown')
    logger.info(f"Найден последний релиз: {tag_name}")

    assets = release_info.get('assets', [])
    
    # Ищем подходящие файлы для скачивания
    for target_folder, match_func in TARGET_MAPPING.items():
        folder_path = RUNTIMES_DIR / target_folder
        
        if not folder_path.exists():
            logger.warning(f"Папка {target_folder} не существует в runtimes/. Пропускаю.")
            continue

        matched_asset = None
        for asset in assets:
            if match_func(asset['name']):
                matched_asset = asset
                break
        
        if matched_asset:
            logger.info(f"[{target_folder}] Найден подходящий архив: {matched_asset['name']}")
            download_and_extract(matched_asset['browser_download_url'], folder_path)
        else:
            logger.warning(f"[{target_folder}] Подходящий архив для этой сборки не найден в релизе {tag_name}.")

    logger.info("Обновление завершено!")

if __name__ == "__main__":
    main()