import json
import logging
from pathlib import Path
import gettext
import os
import locale
from core.utils.logger import setup_logger

class Translator:
    """Класс для локализации текстов"""
    
    def __init__(self, language: str):
        self.language = language
        self.logger = logging.getLogger('AltRAG')
        self.translations = {}
        self._load_translations()
        
    def _load_translations(self):
        """Загружает переводы из JSON-файлов"""
        self.locales_dir = Path(__file__).resolve().parent


        
        try:
            # Загрузка всех доступных языков
            for lang_file in self.locales_dir.glob("*.json"):
                lang_code = lang_file.stem
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            
            self.logger.debug(f"Загружены переводы для: {list(self.translations.keys())}")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки переводов: {str(e)}")
            self.translations = {}
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Возвращает перевод для указанного ключа
        
        Args:
            key: Ключ перевода
            kwargs: Параметры для форматирования строки
            
        Returns:
            Переведенная строка
        """
        try:
            # Пробуем получить перевод для текущего языка
            lang_dict = self.translations.get(self.language, {})
            translation = lang_dict.get(key, key)
            
            # Если есть параметры, форматируем строку
            if kwargs:
                return translation.format(**kwargs)
            return translation
        
        except Exception:
            # В случае ошибки возвращаем оригинальный ключ
            return key