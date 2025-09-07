from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np


class BaseFeature2D(ABC):
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[cv2.KeyPoint]:
        """Найти ключевые точки на изображении"""
        pass

    @abstractmethod
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Посчитать дескрипторы для заданных keypoints"""
        pass

    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """По умолчанию: detect + compute"""
        keypoints = self.detect(image, mask)
        return self.compute(image, keypoints)


class BaseMatcher(ABC):
    @abstractmethod
    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        """Сопоставить два набора дескрипторов"""
        pass
