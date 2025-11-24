from threading import RLock

import numpy as np

from visual_slam.map.observation import Observations

class MapPoint:
    _next_id = 0
    _id_lock = RLock()

    def __init__(
        self, 
        position: np.ndarray, 
        color: np.ndarray = None, 
        descriptor: np.ndarray = None
    ) -> None:
        """
        position: np.ndarray (3,) — координаты точки в мировой системе
        color: np.ndarray (3,) или (B, G, R), если доступен цвет
        descriptor: np.ndarray — дескриптор признака
        """
        with MapPoint._id_lock:
            self.id: int = MapPoint._next_id
            MapPoint._next_id += 1
            
        self._lock = RLock()

        self.position: np.ndarray = np.asarray(position, dtype=float)

        self.color: np.ndarray = color
        self.descriptor: np.ndarray = descriptor

        self.observations = Observations()
        
        self._is_bad: bool = False

    def add_observation(self, kf_idx: int, kp_idx: int) -> bool:
        """
        Добавить наблюдение (KeyFrame ID → KeyPoint index).
        """
        with self._lock:
            return self.observations.add(kf_idx, kp_idx)

    def remove_observation(self, kf_idx: int):
        """Удалить наблюдение по кадру"""
        with self._lock:
            return self.observations.remove(kf_idx)

    def get_observations(self) -> dict[int, int]:
        """Вернуть копию словаря наблюдений."""
        with self._lock:
            return dict(self.observations._data)
        
    def get_observation(self, kf_idx: int) -> int | None:
        """Получить индекс ключевой точки по кадру."""
        with self._lock:
            return self.observations.get(kf_idx)
        
    def clear_observations(self):
        """Очистить все наблюдения."""
        with self._lock:
            self.observations.clear()

    def num_observations(self) -> int:
        """Число наблюдений."""
        with self._lock:
            return len(self.observations)

    def update_position(self, new_position: np.ndarray):
        """Обновить 3D-позицию точки."""
        with self._lock:
            self.position = np.asarray(new_position, dtype=float)
            
    def update_color(self, new_color: np.ndarray):
        """Обновить цвет точки."""
        with self._lock:
            self.color = new_color

    def update_descriptor(self, descs: list[np.ndarray]):
        """
        Обновить дескриптор точки. Можно передавать список дескрипторов из наблюдений.
        Пока берётся первый (можно доработать до медианного/среднего).
        """
        with self._lock:
            if len(descs) == 0:
                return
            self.descriptor = descs[0]

    def mark_as_bad(self):
        """Пометить точку как плохую (например, после фильтрации)."""
        with self._lock:
            self._is_bad = True

    def is_valid(self) -> bool:
        """Проверить пригодность точки."""
        with self._lock:
            if self._is_bad:
                return False
            if not np.isfinite(self.position).all():
                return False
            if self.position[2] <= 0:
                return False
            return True

    def __repr__(self):
        return f"<MapPoint id={self.id}, pos={self.position.round(3)}, obs={len(self.observations)}>"
