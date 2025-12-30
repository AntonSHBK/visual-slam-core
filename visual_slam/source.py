from abc import ABC, abstractmethod
from ast import Tuple
import os
import time
from pathlib import Path
from typing import Union, Tuple, Optional

import cv2
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


class VideoSource:
    def __init__(
        self,
        video_path: Union[str, Path],
        target_fps: float = 5.0,
        use_pos_msec: bool = False,
        cache_frame_shape: bool = True,
        log_dir: str = "logs",
    ):
        self.logger = get_logger(
            self.__class__.__name__,
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO",
        )

        self.video_path = str(video_path)
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"VideoSource: файл не найден: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"VideoSource: не удалось открыть видео: {self.video_path}")

        self.use_pos_msec = use_pos_msec

        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._is_ok = True

        self.target_fps = float(target_fps) if target_fps is not None else 0.0
        self._step = self._compute_step(self._fps, self.target_fps)
        self._cur_idx = 0

        if cache_frame_shape and (self._width == 0 or self._height == 0):
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._height, self._width = frame.shape[:2]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        duration_s = (self._num_frames / self._fps) if self._fps > 1e-8 else None

        self.logger.info("Информация о видео:")
        self.logger.info(f"- Путь: {self.video_path}")
        self.logger.info(f"- Размер кадра: {self._width} x {self._height}")
        self.logger.info(f"- Video FPS: {self._fps:.3f}")
        self.logger.info(f"- Target FPS: {self.target_fps:.3f}")
        self.logger.info(f"- Step (кадров): {self._step}")
        self.logger.info(f"- Число кадров: {self._num_frames}")
        if duration_s is not None:
            self.logger.info(f"- Длительность: {duration_s:.2f} сек")

    @staticmethod
    def _compute_step(video_fps: float, target_fps: float) -> int:
        if target_fps is None or target_fps <= 1e-8:
            return 1
        if video_fps <= 1e-8:
            return 1
        step = int(round(video_fps / target_fps))
        return max(step, 1)

    def set_target_fps(self, target_fps: float):
        self.target_fps = float(target_fps)
        self._step = self._compute_step(self._fps, self.target_fps)
        self.logger.info(f"VideoSource: target_fps={self.target_fps:.3f}, step={self._step}")

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cur_idx = 0
        self._is_ok = True
        self.logger.debug("VideoSource: сброс на начало.")

    def is_ok(self) -> bool:
        if not self._is_ok:
            return False
        if self._num_frames > 0:
            return self._cur_idx < self._num_frames
        return self.cap.isOpened()

    def num_frames(self) -> int:
        return self._num_frames

    def get_frame_shape(self) -> Tuple[int, int, int]:
        if self._width <= 0 or self._height <= 0:
            return (0, 0, 0)
        return (self._height, self._width, 3)

    def get_fps(self) -> float:
        return self._fps

    def get_duration_seconds(self) -> Optional[float]:
        if self._fps > 1e-8 and self._num_frames > 0:
            return self._num_frames / self._fps
        return None

    def seek_seconds(self, t_sec: float) -> bool:
        if self._fps > 1e-8:
            idx = int(round(t_sec * self._fps))
        else:
            idx = int(round(t_sec))
        idx = max(idx, 0)
        if self._num_frames > 0:
            idx = min(idx, self._num_frames - 1)
        ok = bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx))
        if ok:
            self._cur_idx = idx
        return ok

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.logger.info("VideoSource: VideoCapture освобождён.")

    # ------------------------------------------------------------
    # main API
    # ------------------------------------------------------------

    def get_frame(self, idx: Optional[int] = None):
        if not self.cap.isOpened():
            self._is_ok = False
            return None, None

        # --- random access ---
        if idx is not None:
            idx = int(max(idx, 0))
            if self._num_frames > 0 and idx >= self._num_frames:
                return None, None

            ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            if not ok:
                self.logger.warning(f"VideoSource: не удалось выполнить seek на кадр {idx}.")
                return None, None

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._is_ok = False
                self.logger.warning("VideoSource: не удалось прочитать кадр.")
                return None, None

            # при random access логично поставить следующий индекс тоже с шагом
            self._cur_idx = idx + self._step

            timestamp = self._timestamp_from_index(idx)
            return frame, timestamp

        # --- sequential downsample ---
        if self._num_frames > 0 and self._cur_idx >= self._num_frames:
            self._is_ok = False
            return None, None

        # идём сразу на нужный кадр (пропуская промежуточные эффективно)
        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._cur_idx))
        if not ok:
            self.logger.warning(f"VideoSource: не удалось выполнить seek на кадр {self._cur_idx}.")
            self._is_ok = False
            return None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._is_ok = False
            self.logger.warning("VideoSource: не удалось прочитать кадр.")
            return None, None

        idx_out = int(self._cur_idx)
        self._cur_idx += self._step

        timestamp = self._timestamp_from_index(idx_out)
        return frame, timestamp

    def _timestamp_from_index(self, idx: int) -> float:
        """
        timestamp в секундах.
        Важно: для SLAM обычно лучше детерминированно idx/fps.
        """
        if self.use_pos_msec:
            ts_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            return ts_ms / 1000.0
        if self._fps > 1e-8:
            return float(idx) / self._fps
        return float(idx)

    def show(self, window_name: str = "Video", fps: Optional[int] = None):
        """
        Показывает видео с текущим downsample (по target_fps).
        fps здесь — только частота обновления окна, не выбор кадров.
        """
        if fps is None:
            fps = int(self.target_fps) if self.target_fps > 1e-8 else 30
        delay = int(1000 / max(fps, 1))

        self.logger.info(f"VideoSource: показ, window='{window_name}', gui_fps={fps}")
        while self.is_ok():
            frame, ts = self.get_frame()
            if frame is None:
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyWindow(window_name)

    def get_fps(self) -> float:
        return self._fps

    def get_duration_seconds(self) -> Optional[float]:
        if self._fps > 1e-8 and self._num_frames > 0:
            return self._num_frames / self._fps
        return None

    def seek_seconds(self, t_sec: float) -> bool:
        if self._fps > 1e-8:
            idx = int(round(t_sec * self._fps))
            return bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx))
        return bool(self.cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0))

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.logger.info("VideoSource: VideoCapture освобождён.")
