from threading import RLock
from typing import Dict, Optional

import numpy as np
import cv2

from visual_slam.frame import Frame
from visual_slam.pose import Pose
from visual_slam.camera import Camera
from visual_slam.map.map_point import MapPoint


class KeyFrame(Frame):
    _next_id = 0
    _id_lock = RLock()

    def __init__(
        self,
        timestamp: float,
        camera: Camera,
        images: Optional[list[np.ndarray]] = None,
        images_gray: Optional[list[np.ndarray]] = None,
        keypoints: Optional[list[np.ndarray]] = None,
        descriptors: Optional[list[np.ndarray]] = None,
        pose: Optional[Pose] = None,
    ):
        super().__init__(
            timestamp=timestamp,
            camera=camera,
            images=images,
            images_gray=images_gray,
            keypoints=keypoints,
            descriptors=descriptors,
            pose=pose,
        )
        with KeyFrame._id_lock:
            self.keyframe_id = KeyFrame._next_id
            KeyFrame._next_id += 1

        self.map_points: Dict[int, MapPoint] = {}

        self.is_bad: bool = False
        self.is_fixed: bool = False

    @classmethod
    def from_frame(cls, frame: "Frame") -> "KeyFrame":
        images_copy = [img.copy() for img in frame.images] if frame.images else []
        images_gray_copy = [img.copy() for img in frame.images_gray] if frame.images_gray else []

        keypoints_copy = []
        for kps in frame.keypoints:
            if isinstance(kps, (list, tuple)):
                keypoints_copy.append([
                    cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle,
                                kp.response, kp.octave, kp.class_id)
                    for kp in kps
                ])
            else:
                keypoints_copy.append([])

        descriptors_copy = [d.copy() for d in frame.descriptors if d is not None]

        kf = cls(
            timestamp=frame.timestamp,
            camera=frame._camera,
            images=images_copy,
            images_gray=images_gray_copy,
            keypoints=keypoints_copy,
            descriptors=descriptors_copy,
            pose=frame.get_pose().copy(),
        )

        return kf
    
    def add_map_point(self, kp_idx: int, mappoint: MapPoint):
        """Привязать MapPoint к данному KeyFrame по индексу ключевой точки."""
        with self._lock:
            self.map_points[kp_idx] = mappoint
            mappoint.add_observation(self.keyframe_id, kp_idx)

    def get_map_point(self, kp_idx: int) -> Optional[MapPoint]:
        """Получить MapPoint по индексу фичи."""
        return self.map_points.get(kp_idx, None)

    def remove_map_point(self, kp_idx: int):
        """Удалить связь с точкой карты."""
        with self._lock:
            if kp_idx in self.map_points:
                mappoint = self.map_points[kp_idx]
                mappoint.remove_observation(self.keyframe_id)
                del self.map_points[kp_idx]

    def get_all_mappoints(self) -> Dict[int, MapPoint]:
        """Вернуть все точки карты, связанные с этим KeyFrame."""
        with self._lock:
            return dict(self.map_points)

    def mark_as_bad(self):
        """Пометить ключевой кадр как плохой (например, при удалении)."""
        with self._lock:
            self.is_bad = True

    def __repr__(self):
        return (
            f"<KeyFrame id={self.keyframe_id}, "
            f"frame_id={self.id}, "
            f"mps={len(self.map_points)}, "
            f"time={self.timestamp:.3f}>"
        )
