from abc import ABC, abstractmethod
from ast import Tuple
from pathlib import Path

from arrow import get
import cv2
import os
import numpy as np

from visual_slam.utils.logging import get_logger

class DataSourceBase(ABC):

    def __init__(self, log_dir="logs"):
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_ok(self) -> bool:
        pass

    @abstractmethod
    def get_frame(self, idx: int = None):
        pass
    
    @abstractmethod
    def get_frame_shape(self) -> tuple[int, int, int]:
        pass

    @abstractmethod
    def num_frames(self) -> int:
        pass
    
    @abstractmethod
    def show(self, window_name: str = "Frames"):
        pass


class DatasetSource(DataSourceBase):
    def __init__(self, path: str, log_dir: str = "logs"):
        super().__init__(log_dir=log_dir)
        self.path = path
        self.frames = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self._num_frames = len(self.frames)
        self._cur_idx = 0
        
        if self._num_frames > 0:
            img = cv2.imread(self.frames[0], cv2.IMREAD_COLOR)
            frame_height, frame_width = img.shape[:2]
        else:
            frame_width, frame_height = (0, 0)

        self.logger.info("Информация о датасете:")
        self.logger.info(f"* Путь: {self.path}")
        self.logger.info(f"* Размер кадра: {frame_width} x {frame_height}")
        self.logger.info(f"* Общее число кадров: {self._num_frames}")


    def reset(self):
        self._cur_idx = 0
        self.logger.debug("Сброс DatasetSource на начало.")

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
    
    def show(self, window_name: str = "Dataset", fps: int = 30):
        delay = int(1000 / fps)
        while True:
            img, ts = self.get_frame()
            if img is None:
                self.reset()
                continue

            cv2.imshow(window_name, img)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow(window_name)
        
    def get_frame_shape(self) -> tuple[int, int, int]:
        if self._num_frames == 0:
            return (0, 0)
        img = cv2.imread(self.frames[0], cv2.IMREAD_COLOR)
        if img is None:
            return (0, 0)
        return img.shape


class CameraSource(DataSourceBase):
    def __init__(
        self, 
        camera_id: int = 0, 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30, 
        log_dir: str = "logs"
    ):
        super().__init__(log_dir=log_dir)
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._is_ok = self.cap.isOpened()
        self.frame_count = -1
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.logger.info("Информация о камере:")
        self.logger.info(f"- Camera ID: {camera_id}")
        self.logger.info(f"- Размер кадра: {actual_width} x {actual_height}")
        self.logger.info(f"- Запрошенный FPS: {fps}, фактический FPS: {actual_fps:.2f}")
        self.logger.info(f"- Открыта: {self._is_ok}")
        
    def reset(self):
        pass

    def is_ok(self) -> bool:
        return self._is_ok

    def get_frame(self, idx: int = None):
        ret, frame = self.cap.read()
        if not ret:
            self._is_ok = False
            self.logger.warning("Не удалось получить кадр с камеры.")
            return None, None
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        return frame, timestamp

    def num_frames(self) -> int:
        return self.frame_count
    
    def release(self):
        self.cap.release()
        self.logger.info("Камера освобождена.")
        
    def show(self, window_name: str = "Camera"):
        self.logger.info("Запуск отображения живого потока с камеры.")
        while self.is_ok():
            frame, ts = self.get_frame()
            if frame is None:
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        self.release()
        cv2.destroyWindow(window_name)
        
    def get_frame_shape(self) -> tuple[int, int, int]:
        if not self._is_ok:
            return (0, 0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (height, width, 3)
