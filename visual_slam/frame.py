from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict

import numpy as np
import cv2

from visual_slam.map.map_point import MapPoint


@dataclass
class Frame:
    """
    Класс кадра — универсальный контейнер для данных, 
    """

    timestamp: float

    images: List[np.ndarray]
    images_gray: List[np.ndarray] = field(default_factory=list)
    depth: Optional[np.ndarray] = None

    keypoints: List[List[Any]] = field(default_factory=list)
    descriptors: List[np.ndarray] = field(default_factory=list)

    points_3d: Optional[np.ndarray] = None
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))

    map_points: List[Dict[int, MapPoint]] = field(default_factory=list)
    num_visible_mappoints: int = 0

    def __post_init__(self):
        """Автоматически создаёт grayscale версии."""
        if not self.images_gray:
            self.images_gray = []
            for img in self.images:
                if len(img.shape) == 3:
                    self.images_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                else:
                    self.images_gray.append(img.copy())
    
    @property
    def image_left(self):
        return self.images[0] if len(self.images) > 0 else None

    @property
    def image_right(self):
        return self.images[1] if len(self.images) > 1 else None

    @property
    def image_gray_left(self):
        return self.images_gray[0] if len(self.images_gray) > 0 else None

    @property
    def image_gray_right(self):
        return self.images_gray[1] if len(self.images_gray) > 1 else None
    
    @property
    def keypoints_left(self):
        return self.keypoints[0] if len(self.keypoints) > 0 else None

    @property
    def keypoints_right(self):
        return self.keypoints[1] if len(self.keypoints) > 1 else None
    
    @property
    def descriptors_left(self):
        return self.descriptors[0] if len(self.descriptors) > 0 else None
    
    @property
    def descriptors_right(self):
        return self.descriptors[1] if len(self.descriptors) > 1 else None

    def num_keypoints(self) -> int:
        """Количество обнаруженных фич."""
        return len(self.keypoints) if self.keypoints is not None else 0

    def has_3d_points(self) -> bool:
        """Есть ли 3D-точки у кадра."""
        return self.points_3d is not None and len(self.points_3d) > 0

    def set_pose(self, R: np.ndarray, t: np.ndarray):
        """Обновляет матрицу позы по R, t."""
        self.pose = np.eye(4)
        self.pose[:3, :3] = R
        self.pose[:3, 3] = t.reshape(-1)

    def get_pose(self) -> np.ndarray:
        """Возвращает текущую позу (4x4)."""
        return self.pose
    
    def copy(self) -> "Frame":
        return Frame(
            timestamp=self.timestamp,
            images=[img.copy() for img in self.images] if self.images is not None else [],
            images_gray=[img.copy() for img in self.images_gray] if self.images_gray is not None else [],
            depth=self.depth.copy() if isinstance(self.depth, np.ndarray) else self.depth,
            keypoints=[
                [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in kps]
                for kps in self.keypoints
            ] if self.keypoints is not None else [],
            descriptors=[d.copy() for d in self.descriptors] if self.descriptors is not None else [],
            points_3d=self.points_3d.copy() if self.points_3d is not None else None,
            pose=self.pose.copy(),
            map_points=[dict(mps) for mps in self.map_points] if self.map_points is not None else [],
            num_visible_mappoints=self.num_visible_mappoints,
        )

