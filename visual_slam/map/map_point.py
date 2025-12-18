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

    def add_observation(self, kf_idx: int, cam_id: int, kp_idx: int) -> bool:
        with self._lock:
            return self.observations.add(kf_idx, cam_id, kp_idx)

    def remove_observation(self, kf_id: int, cam_id: int | None = None) -> bool:
        with self._lock:
            return self.observations.remove(kf_id, cam_id)

    def get_observations(self) -> dict[int, dict[int, int]]:
        with self._lock:
            return {kf: cam_dict.copy() for kf, cam_dict in self.observations._data.items()}

    def get_observation(self, kf_id: int, cam_id: int | None = None):
        with self._lock:
            return self.observations.get(kf_id, cam_id)
        
    def clear_observations(self):
        with self._lock:
            self.observations.clear()

    def num_observations(self) -> int:
        with self._lock:
            return len(self.observations)

    def update_position(self, new_position: np.ndarray):
        with self._lock:
            self.position = np.asarray(new_position, dtype=float)
            
    def update_color(self, new_color: np.ndarray):
        with self._lock:
            self.color = new_color

    def update_descriptor(self, descs: list[np.ndarray]):
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
