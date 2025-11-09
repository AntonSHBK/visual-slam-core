from ast import List, Set
import threading

from visual_slam.map.map_point import MapPoint
from visual_slam.map.keyframe import KeyFrame
from visual_slam.frame import Frame

class Map:
    """
    Хранилище для 3D-точек и ключевых кадров.
    Управляет добавлением, удалением и доступом к объектам.
    """

    def __init__(self):
        self._points: Set[MapPoint] = set()
        self._keyframes: Set[KeyFrame] = set()
        self._frame: Set[Frame] = set()

        self._lock = threading.RLock()

    def add_map_point(self, mpt: MapPoint):
        """Добавить новую 3D-точку на карту."""
        with self._lock:
            self._points.add(mpt)

    def remove_point(self, mpt: MapPoint):
        """Удалить точку из карты."""
        with self._lock:
            if mpt in self._points:
                self._points.remove(mpt)

    def get_points(self) -> list[MapPoint]:
        """Вернуть список всех 3D-точек."""
        with self._lock:
            return list(self._points)

    def num_points(self) -> int:
        return len(self._points)

    def add_keyframe(self, kf: KeyFrame):
        """Добавить ключевой кадр в карту."""
        with self._lock:
            self._keyframes.add(kf)

    def remove_keyframe(self, kf: KeyFrame):
        """Удалить ключевой кадр."""
        with self._lock:
            if kf in self._keyframes:
                self._keyframes.remove(kf)

    def get_keyframes(self) -> list[KeyFrame]:
        with self._lock:
            return list(self._keyframes)

    def num_keyframes(self) -> int:
        return len(self._keyframes)

    def compute_mean_reproj_error(self, points: List[MapPoint]):
        
        pass
    
    def opimize(self):
        
        pass

    def reset(self):
        """Очистить всю карту (например, при ресете)."""
        with self._lock:
            self._points.clear()
            self._keyframes.clear()

    def __repr__(self):
        return f"<Map | points={len(self._points)}, keyframes={len(self._keyframes)}>"
