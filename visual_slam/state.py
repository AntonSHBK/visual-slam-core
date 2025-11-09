from typing import Optional, Dict


class StateItem:
    __slots__ = ("index", "name", "description", "params")

    def __init__(
        self,
        index: int,
        name: str,
        description: str = "",
        params: Optional[Dict] = None,
    ):
        self.index = index
        self.name = name
        self.description = description
        self.params = params or {}

    def __repr__(self):
        return f"<StateItem {self.name} ({self.index}): {self.description}>"

    def __eq__(self, other):
        if isinstance(other, StateItem):
            return self.index == other.index
        return False

    def __hash__(self):
        return hash(self.index)


class State:
    """
    Класс, содержащий все состояния SLAM/Tracking.
    """

    NO_IMAGES_YET = StateItem(0, "NO_IMAGES_YET", "Нет изображений для обработки")
    NOT_INITIALIZED = StateItem(1, "NOT_INITIALIZED", "Система не инициализирована")
    INITIALIZING = StateItem(2, "INITIALIZING", "Инициализация карты")
    OK = StateItem(3, "OK", "Трекинг в норме")
    LOST = StateItem(4, "LOST", "Трекинг потерян")
    RELOCALIZING = StateItem(5, "RELOCALIZING", "Попытка восстановления позиции")

    MAPPING = StateItem(10, "MAPPING", "Выполняется локальное построение карты")
    LOOP_CLOSING = StateItem(11, "LOOP_CLOSING", "Выполняется замыкание петель")

    @classmethod
    def all_states(cls):
        """
        Вернуть список всех состояний.
        """
        return [
            cls.NO_IMAGES_YET,
            cls.NOT_INITIALIZED,
            cls.INITIALIZING,
            cls.OK,
            cls.LOST,
            cls.RELOCALIZING,
            cls.MAPPING,
            cls.LOOP_CLOSING,
        ]

    @classmethod
    def by_index(cls, idx: int) -> Optional[StateItem]:
        """
        Быстрый доступ по индексу.
        """
        for s in cls.all_states():
            if s.index == idx:
                return s
        return None
