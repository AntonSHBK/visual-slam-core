from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
from threading import RLock

import numpy as np
import cv2

from visual_slam.map.map_point import MapPoint
from visual_slam.pose import Pose
from visual_slam.camera import Camera
from visual_slam.utils.geometry import (
    transform_points_numba
)
from visual_slam.utils.motion_estimation import (
    image_to_gray
)

class FrameBase:
    _id = 0
    _id_lock = RLock()    
    
    def __init__(
        self,
        id: int,
        camera: Camera,
        pose: Pose = None,        
        timestamp: float = 0.0      
    ):
        self._lock_pose = RLock()
        self._camera: Camera = camera
        
        if pose is None: 
            self._pose = Pose() 
        else: 
            self._pose = pose.copy()
            
        if id is not None: 
            self.id = id 
        else: 
            with FrameBase._id_lock:
                self.id = FrameBase._id
                FrameBase._id += 1    
                 
        self.timestamp = timestamp

    @property
    def width(self):
        return self._camera.width

    @property
    def height(self):
        return self._camera.height
    
    @property
    def pose(self) -> Pose:
        with self._lock_pose:
            return self._pose.copy()

    @pose.setter
    def pose(self, new_pose: Pose):
        with self._lock_pose:
            self._pose = new_pose.copy()
    
    @property
    # мир относительно камеры
    def T_w2c(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.T.copy()

    @property
    # камера относительно мира
    def T_c2w(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.inverse().T.copy()

    @property
    def R_w2c(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.R.copy()

    @property
    def R_c2w(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.inverse().R.copy()

    @property
    def t_w2c(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.t.copy()
    
    @property
    def t_c2w(self) -> np.ndarray:
        with self._lock_pose:
            return self._pose.inverse().t.copy()    

    @property
    def quaternion(self) -> np.ndarray:
        """Кватернион ориентации (world→camera)."""
        with self._lock_pose:
            return self._pose.quaternion.copy()
    
    @property
    def euler_rad(self) -> np.ndarray:
        """Эйлеровы углы (roll, pitch, yaw) в радианах."""
        with self._lock_pose:
            return self._pose.euler_rad.copy()

    @property
    def euler_deg(self) -> np.ndarray:
        """Эйлеровы углы (roll, pitch, yaw) в градусах."""
        with self._lock_pose:
            return self._pose.euler_deg.copy()

    # ------------------------------
    # Методы обновления позы, всё в локальной системе
    # ------------------------------

    def update_pose(self, T: np.ndarray):
        with self._lock_pose:
            self._pose.set_matrix(T)


    def update_translation(self, t: np.ndarray):
        with self._lock_pose:
            self._pose.set_translation(t)


    def update_rotation(self, R: np.ndarray):
        with self._lock_pose:
            self._pose.set_rotation(R)


    def update_rotation_and_translation(
        self,
        R: np.ndarray,
        t: np.ndarray,
    ):
        with self._lock_pose:
            self._pose.set_RT(R, t)

    # ------------------------------
    # Преобразования точек
    # ------------------------------

    def transform_point(self, point_w: np.ndarray) -> np.ndarray:
        """Преобразовать точку в глобальной СК в координаты камеры."""
        with self._lock_pose:
            return self.R_w2c @ point_w + self.t_w2c

    def transform_points(self, points_w: np.ndarray) -> np.ndarray:
        """Преобразовать массив точек (Nx3) в систему координат камеры."""
        with self._lock_pose:
            Rcw = self.R_w2c
            tcw = self.t_w2c
        points_w = np.ascontiguousarray(points_w)
        
        if points_w.shape[0] < 500:
            return (Rcw @ points_w.T + tcw.reshape(3, 1)).T
        else:
            return transform_points_numba(points_w, Rcw, tcw)

    # ------------------------------
    # Проекция 3D→2D
    # ------------------------------

    def project_point(self, point_w: np.ndarray):
        """Проекция одной 3D точки в изображение."""
        point_c = self.transform_point(point_w)
        return self._camera.project(point_c)

    def project_points(self, points: np.ndarray):
        """Проекция массива 3D точек (Nx3) в изображение."""
        points_c = self.transform_points(points)
        return self._camera.project(points_c)

    # ------------------------------
    # Проверки на видимость
    # ------------------------------

    def is_in_image(self, uv, z) -> bool:
        """Проверить, попадает ли точка в изображение."""
        return self._camera.is_in_image(uv, z)

    def are_in_image(self, uvs, zs):
        """Пакетная проверка массива точек на попадание в изображение."""
        return self._camera.are_in_image(uvs, zs)
    
    # ------------------------------
    # Проверки видимости 3D-точек
    # ------------------------------

    def is_visible(
        self, 
        map_point: "MapPoint"
    ) -> tuple[bool, np.ndarray, float]:
        """
        Проверяет, видна ли 3D-точка map_point в данном кадре.
        """
        uv, z = self.project_point(map_point.position)
        if not self.is_in_image(uv, z) or z <= 0:
            return False, uv, z

        # Центр камеры (Ow) в мировой системе
        with self._lock_pose:
            Ow = self.t_c2w

        # Вектор от камеры к точке
        PO = map_point.position - Ow
        dist3D = np.linalg.norm(PO)
        if dist3D < 1e-6 or not np.isfinite(dist3D):
            return False, uv, z

        # Вычисляем нормаль как направление "взгляда" (камера → точка)
        normal = PO / dist3D

        # Проверяем угол между направлением камеры (ось Z в СК камеры)
        # и направлением на точку (в мировой СК)
        # Для этого получаем направление взгляда камеры в мировой системе:
        with self._lock_pose:
            Rwc = self.R_c2w
            view_dir = Rwc[:, 2]

        cos_view = np.dot(view_dir, normal)

        # Проверка угла видимости (например, < 60°)
        if cos_view < 0.5:  # cos(60°)
            return False, uv, z

        return True, uv, z

    def are_visible(
        self, 
        map_points: list["MapPoint"]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Проверяет видимость набора 3D-точек относительно текущего кадра.
        Нормали вычисляются как направления от камеры к точкам.
        """
        if len(map_points) == 0:
            return np.array([], dtype=bool), np.empty((0, 2)), np.array([]), np.array([])

        points = np.vstack([p.position for p in map_points])
        uvs, zs = self.project_points(points)

        with self._lock_pose:
            Ow = self.t_c2w
            Rwc = self.R_c2w
            view_dir = Rwc[:, 2]

        # Векторы от камеры к точкам
        POs = points - Ow
        dists = np.linalg.norm(POs, axis=1)
        valid_mask = (dists > 1e-6) & np.isfinite(dists)

        # Нормали (единичные направления)
        normals = np.zeros_like(POs)
        normals[valid_mask] = POs[valid_mask] / dists[valid_mask, None]

        # Косинусы углов между направлением взгляда камеры и нормалями
        cos_view = normals @ view_dir

        are_in_image = self.are_in_image(uvs, zs)
        are_in_front = zs > 0
        are_in_angle = cos_view > 0.5  # угол меньше 60°

        flags = are_in_image & are_in_front & are_in_angle & valid_mask

        return flags, uvs, zs, dists

   
class Frame(FrameBase):
    def __init__(
        self,
        timestamp: float,
        camera: Camera,
        images: List[np.ndarray] = None,  
        images_gray: List[np.ndarray] = None,      
        keypoints:List[List[Any]] = None,
        descriptors: List[np.ndarray] = None,
        pose: Pose = None,
    ):
        super().__init__(
            id=None, 
            camera=camera, 
            pose=pose, 
            timestamp=timestamp
        )

        self._lock = RLock()

        self.images: List[np.ndarray] = images or []
        self.images_gray: List[np.ndarray] = images_gray or []

        self.keypoints: List[List[Any]] = keypoints or []
        self.descriptors: List[np.ndarray] = descriptors or []

        if len(self.images_gray) == 0:
            for img in self.images:
                if len(img.shape) == 3:
                    self.images_gray.append(image_to_gray(img))
                else:
                    self.images_gray.append(img.copy())

    # ------------------------------
    # Свойства изображений
    # ------------------------------

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

    # ------------------------------
    # Фичи и дескрипторы
    # ------------------------------

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
    
    def set_keypoints(self, value: List[List[Any]]):
        with self._lock:
            self._keypoints = value 
            
    def set_descriptors(self, value: List[np.ndarray]):
        with self._lock:
            self._descriptors = value

    def num_keypoints(self) -> int:
        """Количество ключевых точек (по всем изображениям)."""
        return sum(len(kps) for kps in self.keypoints) if self.keypoints else 0

    def get_keypoints(self, cam_idx: int = 0) -> list:
        """Вернуть keypoints для заданной камеры."""
        if not self.keypoints or cam_idx >= len(self.keypoints):
            return []
        return self.keypoints[cam_idx]

    def get_descriptors(self, cam_idx: int = 0) -> np.ndarray:
        """Вернуть дескрипторы для заданной камеры."""
        if not self.descriptors or cam_idx >= len(self.descriptors):
            return np.empty((0, 32), dtype=np.uint8)
        return self.descriptors[cam_idx]
    
    def get_image(self, cam_idx: int = 0) -> np.ndarray:
        """Вернуть изображение для заданной камеры."""
        if not self.images or cam_idx >= len(self.images):
            return None
        return self.images[cam_idx]
    
    def get_image_gray(self, cam_idx: int = 0) -> np.ndarray:
        """Вернуть серое изображение для заданной камеры."""
        if not self.images_gray or cam_idx >= len(self.images_gray):
            return None
        return self.images_gray[cam_idx]
    
    # ------------------------------
    # Работа с позой
    # ------------------------------
    
    def set_pose(self, T: np.ndarray):
        with self._lock_pose:
            self._pose.set_matrix(T)

    def set_pose_Rt(self, R: np.ndarray, t: np.ndarray):
        with self._lock_pose:
            self._pose.set_RT(R, t)

    def get_pose(self) -> Pose:
        """Возвращает объект Pose."""
        with self._lock_pose:
            return self._pose.copy()

    def get_pose_matrix(self) -> np.ndarray:
        """Возвращает SE(3) матрицу 4x4."""
        with self._lock_pose:
            return self._pose.as_matrix()

    def __repr__(self):
        return f"<Frame id={self.id}, kps={self.num_keypoints()}, time={self.timestamp:.3f}, pose t={self.t_w2c.round(3)}, R=\n{self.R_w2c.round(3)}>"