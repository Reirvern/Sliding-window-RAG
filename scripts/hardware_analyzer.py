# scripts/hardware_analyzer.py
import subprocess
import os
import ctypes
import logging

logger = logging.getLogger('HardwareAnalyzer')

def get_system_ram_gb() -> float:
    """Возвращает объем оперативной памяти в ГБ."""
    try:
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', c_ulonglong),
                ('ullAvailPhys', c_ulonglong),
                ('ullTotalPageFile', c_ulonglong),
                ('ullAvailPageFile', c_ulonglong),
                ('ullTotalVirtual', c_ulonglong),
                ('ullAvailVirtual', c_ulonglong),
                ('ullAvailExtendedVirtual', c_ulonglong),
            ]
        memoryStatus = MEMORYSTATUSEX()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
        return memoryStatus.ullTotalPhys / (1024**3)
    except Exception:
        return 0.0

def get_gpu_vram_gb() -> float:
    """Пытается получить объем VRAM NVIDIA в ГБ."""
    try:
        # Запрашиваем объем памяти видеокарты через nvidia-smi
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
            encoding='utf-8'
        )
        mb = sum(int(x.strip()) for x in output.strip().split('\n'))
        return mb / 1024.0
    except Exception:
        return 0.0

def analyze_hardware() -> dict:
    """Собирает отчет о железе."""
    ram = get_system_ram_gb()
    vram = get_gpu_vram_gb()
    threads = os.cpu_count() or 1

    recommendation = "CPU-инференс"
    if vram >= 12:
        recommendation = "CUDA (Большие модели 8B-14B)"
    elif vram >= 6:
        recommendation = "CUDA (Средние модели 4B-7B)"
    elif vram > 0:
        recommendation = "CUDA (Легкие модели 2B-3B)"

    return {
        "ram_gb": round(ram, 1),
        "vram_gb": round(vram, 1),
        "cpu_threads": threads,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    report = analyze_hardware()
    print("=== Отчет о железе ===")
    print(f"RAM: {report['ram_gb']} ГБ")
    print(f"VRAM (NVIDIA): {report['vram_gb']} ГБ")
    print(f"Потоки CPU: {report['cpu_threads']}")
    print(f"Рекомендация системы: {report['recommendation']}")