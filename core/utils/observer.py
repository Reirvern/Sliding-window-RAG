# core/utils/observer.py
from typing import Protocol, List, Any

class Observer(Protocol):
    """Интерфейс для наблюдателя."""
    def update(self, message_type: str, data: Any):
        """
        Метод, вызываемый Observable для уведомления наблюдателей.
        :param message_type: Тип сообщения (например, "progress", "complete", "error").
        :param data: Данные, связанные с сообщением (например, текущий прогресс, результат).
        """
        pass

class Observable:
    """Базовый класс для объектов, которые могут быть наблюдаемыми."""
    def __init__(self):
        self._observers: List[Observer] = []

    def add_observer(self, observer: Observer):
        """Добавляет наблюдателя."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        """Удаляет наблюдателя."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, message_type: str, data: Any):
        """Уведомляет всех наблюдателей."""
        for observer in self._observers:
            observer.update(message_type, data)