class InterfaceFactory:
    @staticmethod
    def create_interface(interface_type: str):
        """
        Создает и возвращает объект интерфейса по его типу
        
        :param interface_type: Тип интерфейса ('cli', 'gui' и т.д.)
        :return: Объект интерфейса
        :raises ValueError: Если передан неизвестный тип интерфейса
        """
        if interface_type == "cli":
            from .cli import CLIInterface
            return CLIInterface()
        else:
            # Позже можно будет добавить другие интерфейсы
            raise ValueError(f"Неизвестный тип интерфейса: {interface_type}")