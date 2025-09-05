from abc import ABC, abstractmethod

class BaseFeature2D(ABC):
    """Базовый интерфейс для детектора+дескриптора"""

    @abstractmethod
    def detect(self, image, mask=None):
        """Найти ключевые точки на изображении"""
        pass

    @abstractmethod
    def compute(self, image, keypoints):
        """Посчитать дескрипторы для заданных keypoints"""
        pass

    def detectAndCompute(self, image, mask=None):
        """По умолчанию: detect + compute"""
        keypoints = self.detect(image, mask)
        return self.compute(image, keypoints)


class BaseMatcher(ABC):
    """Базовый интерфейс для матчеров"""

    @abstractmethod
    def match(self, desc1, desc2):
        """Сопоставить два набора дескрипторов"""
        pass
