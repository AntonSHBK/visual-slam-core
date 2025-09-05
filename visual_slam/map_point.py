import numpy as np

class MapPoint:
    _next_id = 0  # глобальный счётчик id

    def __init__(self, position, color=None, descriptor=None):
        """
        position: np.ndarray (3,) — координаты точки в мировой системе
        color: np.ndarray (3,) или (B, G, R), если доступен цвет
        descriptor: np.ndarray — дескриптор признака
        """
        self.id = MapPoint._next_id
        MapPoint._next_id += 1

        # Геометрия
        self.position = np.asarray(position, dtype=float)
        self.normal = np.array([0, 0, 1], dtype=float)

        # Визуальная информация
        self.color = color
        self.descriptor = descriptor

        # Наблюдения
        self.observations = {}   # keyframe -> keypoint index
        self.num_visible = 0     # сколько раз точка попадала в поле зрения
        self.num_tracked = 0     # сколько раз реально заматчена

        # Состояние
        self.is_bad = False
        self.reference_keyframe = None

    # --------- Методы работы с наблюдениями ---------

    def add_observation(self, keyframe, kp_idx):
        """Добавить наблюдение из keyframe с индексом ключевой точки."""
        if keyframe not in self.observations:
            self.observations[keyframe] = kp_idx
            self.num_visible += 1
            return True
        return False

    def remove_observation(self, keyframe):
        """Удалить наблюдение из keyframe."""
        if keyframe in self.observations:
            del self.observations[keyframe]
        if len(self.observations) < 2:
            self.is_bad = True

    def get_observations(self):
        return list(self.observations.items())

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
            # простая стратегия: взять первый
            self.descriptor = descs[0]

    # --------- Утилиты ---------

    def get_found_ratio(self):
        """Вернуть отношение успешно заматченных к видимым."""
        return 0 if self.num_visible == 0 else self.num_tracked / self.num_visible

    def __repr__(self):
        return f"<MapPoint id={self.id}, pos={self.position}, obs={len(self.observations)}>"
