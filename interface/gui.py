import logging
from core.utils.localization.translator import Translator  # Добавляем импорт

class GUIInterface:
    def __init__(self, config: dict, rag_engine, logger, translator: Translator):
        self.config = config
        self.rag_engine = rag_engine
        self.logger = logger
        self.translator = translator
    
    def run(self):
        self.logger.info("GUI интерфейс пока не реализован")
        print("GUI интерфейс находится в разработке")