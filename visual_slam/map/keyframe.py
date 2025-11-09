import numpy as np


class KeyFrame:
    _next_id = 0

    def __init__(
        self,
        pose: np.ndarray,
        keypoints: list,
        descriptors: np.ndarray,
        timestamp: float = None,
        image: np.ndarray = None,
    ) -> None:
        """
        pose: np.ndarray (4x4) — матрица преобразования камеры в мировой системе
        keypoints: list[cv2.KeyPoint] или np.ndarray — список ключевых точек
        descriptors: np.ndarray — дескрипторы (N x D)
        timestamp: float — время съемки
        image: np.ndarray | None — изображение (опционально, например, в debug-режиме)
        """
        self.id: int = KeyFrame._next_id
        KeyFrame._next_id += 1

        self.pose: np.ndarray = pose.astype(float)      # T_wc (камера в мировой системе)
        self.keypoints: list = keypoints
        self.descriptors: np.ndarray = descriptors
        self.timestamp: float = timestamp
        self._image: np.ndarray | None = image
        self.points: dict = {}
        self.is_bad: bool = False
        
    def add_image(self, image: np.ndarray):
        self._image = image.copy()

    @property
    def image(self) -> np.ndarray:
        """
        Вернуть изображение.
        Если изображение отсутствует, возвращается "пустое" серое изображение 10x10.
        """
        if self._image is None:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        return self._image

    @image.setter
    def image(self, value: np.ndarray | None):
        """Позволяет установить изображение напрямую (при необходимости)."""
        self._image = value

    def add_point_match(self, mpt, idx):
        """Привязать MapPoint к keypoint с индексом idx."""
        self.points[idx] = mpt

    def remove_point_match(self, idx):
        """Удалить связь keypoint -> MapPoint."""
        if idx in self.points:
            del self.points[idx]

    def get_point_match(self, idx):
        """Вернуть MapPoint по индексу keypoint."""
        return self.points.get(idx, None)

    def get_points(self):
        """Вернуть все MapPoints, привязанные к этому KeyFrame."""
        return list(self.points.values())

    def set_pose(self, pose: np.ndarray):
        """Обновить позу камеры."""
        self.pose = pose.astype(float)

    def get_camera_center(self):
        """Вернуть координаты центра камеры (Ow)."""
        return self.pose[:3, 3]

    def __repr__(self):
        return f"<KeyFrame id={self.id}, kps={len(self.keypoints)}, mpts={len(self.points)}>"
