import numpy as np

from visual_slam.map.observation import Observations, Observation

class MapPoint:
    _next_id = 0

    def __init__(
        self, 
        position: np.ndarray, 
        color: np.ndarray | None = None, 
        descriptor: np.ndarray | None = None
    ) -> None:
        """
        position: np.ndarray (3,) — координаты точки в мировой системе
        color: np.ndarray (3,) или (B, G, R), если доступен цвет
        descriptor: np.ndarray — дескриптор признака
        """
        self.id: int = MapPoint._next_id
        MapPoint._next_id += 1

        # Геометрия
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.normal: np.ndarray = np.array([0, 0, 1], dtype=float)

        # Визуальная информация
        self.color: np.ndarray | None = color
        self.descriptor: np.ndarray | None = descriptor

        # Наблюдения
        self.observations = Observations()
        self.num_visible: int = 0     # сколько раз точка попадала в поле зрения
        self.num_tracked: int = 0     # сколько раз заматчена

        # Состояние
        self._is_bad: bool = False
        self.reference_keyframe: object | None = None

    # --------- Методы работы с наблюдениями ---------

    def add_observation(self, kf_idx: int, kp_idx: int, uv=None, scale=None):
        obs = Observation(kf_idx, kp_idx, uv, scale)
        if self.observations.add(obs):
            self.num_visible += 1
            return True
        return False

    def remove_observation(self, kf_idx: int):
        self.observations.remove(kf_idx)
        if len(self.observations) < 2:
            self._is_bad = True

    def get_observations(self):
        return self.observations.all()


    # --------- Методы обновления информации ---------

    def update_position(self, new_position):
        """Обновить 3D-позицию точки."""
        self.position = np.asarray(new_position, dtype=float)

    def update_normal(self):
        """Пересчитать нормаль как среднее направление от камер к точке."""
        if not self.observations:
            return
        directions = []
        for kf in self.observations.keys():
            cam_pos = kf.pose[:3, 3]  # камера в мировой системе
            v = self.position - cam_pos
            v /= np.linalg.norm(v)
            directions.append(v)
        self.normal = np.mean(directions, axis=0)
        self.normal /= np.linalg.norm(self.normal)

    def update_descriptor(self):
        """Выбрать лучший дескриптор среди наблюдений (например, медианный)."""
        if not self.observations:
            return
        descs = [kf.descriptors[idx] for kf, idx in self.observations.items()]
        if len(descs) == 1:
            self.descriptor = descs[0]
        else:
            self.descriptor = descs[0]
            
    def mark_as_bad(self):
        """Пометить точку как плохую (например, после BA или фильтрации)."""
        self._is_bad = True

    def is_valid(self) -> bool:
        """Возвращает True, если точка пригодна для использования."""
        if self._is_bad:
            return False
        if not np.isfinite(self.position).all():
            return False
        if self.position[2] <= 0:
            return False
        return True

    # --------- Утилиты ---------

    def get_found_ratio(self):
        """Вернуть отношение успешно заматченных к видимым."""
        return 0 if self.num_visible == 0 else self.num_tracked / self.num_visible

    def __repr__(self):
        return f"<MapPoint id={self.id}, pos={self.position}, obs={len(self.observations)}>"
