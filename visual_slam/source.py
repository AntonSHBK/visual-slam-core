import cv2
import os
import numpy as np
from abc import ABC, abstractmethod


class DataSourceBase(ABC):
    """
    Базовый класс для источников данных (датасет или живая камера).
    """

    @abstractmethod
    def reset(self):
        """Сброс источника (например, начать воспроизведение с начала)."""
        pass

    @abstractmethod
    def is_ok(self) -> bool:
        """Возвращает True, если источник готов отдавать кадры."""
        pass

    @abstractmethod
    def get_frame(self, idx: int = None):
        """
        Получить кадр по индексу (если оффлайн) или следующий (если поток).
        Должен вернуть (изображение RGB или BGR, timestamp).
        """
        pass

    @abstractmethod
    def num_frames(self) -> int:
        """Вернуть количество кадров, если доступно (иначе -1)."""
        pass


class DatasetSource(DataSourceBase):
    """
    Класс для работы с оффлайн-датасетами (например, набор картинок или видеофайл).
    """

    def __init__(self, path: str):
        self.path = path
        self.frames = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self._num_frames = len(self.frames)
        self._cur_idx = 0

    def reset(self):
        self._cur_idx = 0

    def is_ok(self) -> bool:
        return self._cur_idx < self._num_frames

    def get_frame(self, idx: int = None):
        if idx is None:
            idx = self._cur_idx
            self._cur_idx += 1
        if idx >= self._num_frames:
            return None, None

        img = cv2.imread(self.frames[idx], cv2.IMREAD_COLOR)
        timestamp = idx
        return img, timestamp

    def num_frames(self) -> int:
        return self._num_frames


class CameraSource(DataSourceBase):
    """
    Класс для работы с живой камерой (cv2.VideoCapture).
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._is_ok = self.cap.isOpened()
        self.frame_count = -1  # для живого потока количество кадров заранее неизвестно

    def reset(self):
        # Для реальной камеры reset обычно не нужен
        pass

    def is_ok(self) -> bool:
        return self._is_ok

    def get_frame(self, idx: int = None):
        ret, frame = self.cap.read()
        if not ret:
            self._is_ok = False
            return None, None
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        return frame, timestamp

    def num_frames(self) -> int:
        return self.frame_count  # -1 => неизвестно (стрим)

    def release(self):
        self.cap.release()
