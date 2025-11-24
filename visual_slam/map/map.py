from typing import Deque, Set, List
from collections import deque
from threading import RLock

import numpy as np

from visual_slam.map.map_point import MapPoint
from visual_slam.map.keyframe import KeyFrame
from visual_slam.frame import Frame
from visual_slam.optimization.base_optimizer import BaseOptimizer
from visual_slam.camera import Camera
from visual_slam.utils.geometry import compute_reprojection_error

class Map:
    def __init__(self):
        self._points: Set[MapPoint] = set()
        self._keyframes: Set[KeyFrame] = set()        
        self.frames: Deque[Frame] = deque(maxlen=5)

        self._lock = RLock()

    def add_map_point(self, mpt: MapPoint):
        with self._lock:
            self._points.add(mpt)

    def remove_point(self, mpt: MapPoint):
        with self._lock:
            if mpt in self._points:
                self._points.remove(mpt)

    def get_points(self) -> list[MapPoint]:
        with self._lock:
            return list(self._points)

    def num_points(self) -> int:
        return len(self._points)

    def add_keyframe(self, kf: KeyFrame):
        with self._lock:
            self._keyframes.add(kf)

    def remove_keyframe(self, kf: KeyFrame):
        with self._lock:
            if kf in self._keyframes:
                self._keyframes.remove(kf)

    def get_keyframes(self) -> list[KeyFrame]:
        with self._lock:
            return list(self._keyframes)

    def num_keyframes(self) -> int:
        return len(self._keyframes)
    
    def compute_mean_reprojection_error(
        self,
        camera: "Camera",
        keyframes: list["KeyFrame"] = None,
    ) -> float:

        keyframes = keyframes or self.get_keyframes()
        points = self.get_points()
        if len(keyframes) == 0 or len(points) == 0:
            return 0.0

        total_error = 0.0
        total_count = 0

        for kf in keyframes:
            pts_3d = []
            pts_2d = []

            for idx, mp in kf.map_points.items():
                if mp is None or not mp.is_valid():
                    continue
                pts_3d.append(mp.position)
                kp_list = kf.get_keypoints()
                if idx >= len(kp_list):
                    continue
                pts_2d.append(kp_list[idx].pt)

            if len(pts_3d) == 0 or len(pts_2d) == 0:
                continue

            pts_3d = np.array(pts_3d, dtype=float)
            pts_2d = np.array(pts_2d, dtype=float)

            _, mean_err = compute_reprojection_error(
                points_3d=pts_3d,
                points_2d=pts_2d,
                K=camera.K,
                R=kf.Rcw,
                t=kf.tcw
            )
            total_error += mean_err * len(pts_3d)
            total_count += len(pts_3d)

        mean_error = total_error / max(total_count, 1)

        return mean_error

    def optimize_initial(self, optimizer: BaseOptimizer) -> bool:
        keyframes = self.get_keyframes()
        points = self.get_points()
        if len(keyframes) < 2 or len(points) == 0:
            return False
        return optimizer.optimize_initial(keyframes, points)

    def optimize_local(self, optimizer: BaseOptimizer) -> bool:
        keyframes = self.get_keyframes()
        points = self.get_points()
        if len(keyframes) == 0 or len(points) == 0:
            return False
        return optimizer.optimize_local(keyframes, points)

    def optimize_global(self, optimizer: BaseOptimizer) -> bool:
        keyframes = self.get_keyframes()
        points = self.get_points()
        if len(keyframes) == 0 or len(points) == 0:
            return False
        return optimizer.optimize_global(keyframes, points)

    # def optimize(self, optimizer: BaseOptimizer, mode: str = "local") -> bool:
    #     """
    #     Унифицированный интерфейс вызова оптимизации.
    #     """
    #     if mode == "initial":
    #         return self.optimize_initial(optimizer)
    #     elif mode == "global":
    #         return self.optimize_global(optimizer)
    #     elif mode == "local":
    #         return self.optimize_local(optimizer)
    #     else:
    #         raise ValueError(f"[Map] Неизвестный режим оптимизации: {mode}")

    def reset(self):
        with self._lock:
            self._points.clear()
            self._keyframes.clear()

    def __repr__(self):
        return f"<Map | points={len(self._points)}, keyframes={len(self._keyframes)}>"
