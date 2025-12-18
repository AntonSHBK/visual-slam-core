from typing import Deque, Set, List
from collections import deque
from threading import RLock

import numpy as np

from visual_slam.map.map_point import MapPoint
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.frame import Frame
from visual_slam.optimization.base_optimizer import BaseOptimizer
from visual_slam.camera import Camera
from visual_slam.utils.geometry import compute_reprojection_error

class Map:
    def __init__(
        self,
        max_frames: int = 5,
    ):
        self._points: Set[MapPoint] = set()
        self._keyframes: Deque[KeyFrame] = deque()       
        self._frames: Deque[Frame] = deque(maxlen=max_frames)
        self._lock = RLock()
        
    # ================ MapPoint Management ===================

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
    
    # ================ Frame Management ===================
    
    def add_frame(self, frame: Frame):
        with self._lock:
            self._frames.append(frame)
            
    def get_frames(self) -> List[Frame]:
        with self._lock:
            return list(self._frames)    
    
    def get_first_frame(self) -> Frame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[0]

    def get_last_frame(self) -> Frame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    
    # ================ KeyFrame Management ===================
        
    def add_keyframe(self, kf):
        with self._lock:
            self._keyframes.append(kf)

    def remove_keyframe(self, kf: KeyFrame):
        with self._lock:
            if kf in self._keyframes:
                self._keyframes.remove(kf)

    def get_keyframes(self) -> list[KeyFrame]:
        with self._lock:
            return list(self._keyframes)
    
    def get_last_keyframe(self):
        with self._lock:
            if not self._keyframes:
                return None
            return self._keyframes[-1]
    
    def get_first_keyframe(self):
        with self._lock:
            if not self._keyframes:
                return None
            return self._keyframes[0]
        
    def num_keyframes(self) -> int:
        return len(self._keyframes)
    
    # ================ Optimization and Error Computation ===================
    
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

            for (cam_id, kp_idx), mp in kf.map_points.items():
                if mp is None or not mp.is_valid():
                    continue
                pts_3d.append(mp.position)
                kps_for_cam = kf.keypoints[cam_id]

                if kp_idx >= len(kps_for_cam):
                    continue
                pts_2d.append(kps_for_cam[kp_idx].pt)
            if len(pts_3d) == 0 or len(pts_2d) == 0:
                continue

            pts_3d = np.array(pts_3d, dtype=float)
            pts_2d = np.array(pts_2d, dtype=float)

            _, mean_err = compute_reprojection_error(
                points_3d=pts_3d,
                points_2d=pts_2d,
                K=camera.K,
                R=kf.R_w2c,
                t=kf.t_w2c
            )

            total_error += mean_err * len(pts_3d)
            total_count += len(pts_3d)

        return total_error / max(total_count, 1)

    def optimize_initial(self, optimizer: BaseOptimizer) -> bool:
        keyframes = self.get_keyframes()
        points = self.get_points()
        if len(keyframes) < 2 or len(points) == 0:
            return False
        return optimizer.optimize_initial(keyframes, points)

    def optimize_local(
        self,
        optimizer: BaseOptimizer,
        keyframes: List[KeyFrame]
    ) -> bool:

        if not keyframes:
            return False

        points_raw = []
        for kf in keyframes:
            mp_dict = kf.get_all_mappoints()
            for mp in mp_dict.values():
                points_raw.append(mp)

        unique_points = {}
        for mp in points_raw:
            unique_points[id(mp)] = mp

        points = list(unique_points.values())
        
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
    
    def normalize_scale(self):
        # TODO реализовать нормализацию масштаба карты с учётом frame и keyframe
        pass

    def reset(self):
        with self._lock:
            self._points.clear()
            self._keyframes.clear()
            self._frames.clear()

    def __repr__(self):
        return f"<Map | points={len(self._points)}, keyframes={len(self._keyframes)}>"
