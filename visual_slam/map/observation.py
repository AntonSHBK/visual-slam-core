from typing import Dict, Optional


class Observations:
    """
    Хранит наблюдения точки по кадрам (keyframe_id → keypoint_index).
    Использует словарь для быстрого доступа и модификации (O(1)).
    """

    __slots__ = ("_data",)

    def __init__(self):
        self._data: Dict[int, int] = {}  # key: kf_idx, value: kp_idx

    # ------------------------------
    # Базовые операции
    # ------------------------------

    def add(self, kf_idx: int, kp_idx: int) -> bool:
        """Добавить наблюдение. Возвращает False, если уже есть."""
        if kf_idx not in self._data:
            self._data[kf_idx] = kp_idx
            return True
        return False

    def remove(self, kf_idx: int) -> bool:
        """Удалить наблюдение по keyframe."""
        try:
            self._data.pop(kf_idx)
            return True
        except KeyError:
            return False

    def get(self, kf_idx: int) -> Optional[int]:
        """Получить индекс ключевой точки по кадру."""
        return self._data.get(kf_idx, None)

    def has(self, kf_idx: int) -> bool:
        """Проверить наличие наблюдения для кадра."""
        return kf_idx in self._data

    def clear(self):
        """Очистить все наблюдения."""
        self._data.clear()

    # ------------------------------
    # Итерация и служебные методы
    # ------------------------------

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        """Итерировать как (kf_idx, kp_idx)."""
        return iter(self._data.items())

    def keyframes(self):
        """Список id всех keyframes."""
        return list(self._data.keys())

    def keypoints(self):
        """Список индексов всех ключевых точек."""
        return list(self._data.values())

    def __repr__(self):
        return f"<Observations n={len(self._data)}>"
