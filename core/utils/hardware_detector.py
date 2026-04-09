# core/utils/hardware_detector.py
import subprocess
import logging

logger = logging.getLogger('AltRAG')

def detect_best_runtime() -> str:
    """
    Пытается определить лучшее доступное железо и возвращает название папки рантайма.
    """
    # 1. Ищем NVIDIA GPU (CUDA)
    try:
        # nvidia-smi - стандартная утилита драйверов NVIDIA
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        logger.info("Обнаружена видеокарта NVIDIA. Рекомендуется CUDA 12.")
        return "cuda12" # По умолчанию ставим 12, так как 13 еще мало у кого есть
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 2. Ищем AMD GPU (здесь сложнее, но можно проверить Vulkan)
    # В Windows можно попытаться найти vulkaninfo
    try:
        subprocess.check_output(["vulkaninfo"], stderr=subprocess.STDOUT)
        logger.info("Обнаружена поддержка Vulkan.")
        return "vulkan"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 3. Фолбэк на процессор
    logger.info("Дискретная видеокарта не обнаружена. Используем CPU.")
    return "cpu_x64"

def get_available_runtimes(runtimes_dir: str) -> list:
    """Возвращает список папок внутри директории runtimes"""
    from pathlib import Path
    p = Path(runtimes_dir)
    if not p.exists():
        return []
    return [d.name for d in p.iterdir() if d.is_dir()]