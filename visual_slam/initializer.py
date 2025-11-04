from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Any, Deque

import cv2
import numpy as np
from sympy import im

from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint
from visual_slam.utils.logging import get_logger
from visual_slam.utils.geometry import (
    poseRt, 
    inv_poseRt, 
    rpy_from_rotation_matrix
)
from visual_slam.utils.motion_estimation import (
        estimate_motion_from_2d2d,
        triangulate_points,
        triangulate_normalized_points,
        compute_baseline,
        compute_parallax,
        compute_normalize_parallax,
        filter_by_parallax,
        check_feature_coverage,
        normalize_depth_scale,
        triangulate_stereo_points,
        image_to_gray
    )

from visual_slam.utils.geometry import normalize
from visual_slam.viz.feature_viz import FeatureVisualizer
from visual_slam.frame import Frame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.tracking import Tracking
    from visual_slam.config import Config
    from visual_slam.feature.tracker import FeatureTrackingResult


class Initializer:
    def __init__(
        self, 
        tracking: "Tracking", 
        config: "Config",
        log_dir="logs"
    ):
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )
        
        self.tracking = tracking
        self.config = config

        self.min_inliers = self.config.tracking.min_inliers
        self.camera = self.tracking.camera
        
        self.frames: Deque[Frame] = deque(maxlen=5)
        
        self.initialized: bool = False
        self.num_failures = 0
        
        self.viz = FeatureVisualizer(window_name_prefix="Initializer")

        self.logger.info("\n\n[Initializer] Инициализация ====================>\n")

    def reset(self):
        self.frames.clear()
        self.initialized = False
        self.min_inliers = self.config.tracking.min_inliers
        self.num_failures = 0
        self.logger.info("[Initializer] Сброс инициализатора.")

    def add_frame(self, images: List[np.ndarray], timestamp: float):
        images_gray = [image_to_gray(img) for img in images]

        keypoints_list = []
        descriptors_list = []

        for i, img_gray in enumerate(images_gray):
            kps, des = self.tracking.feature_tracker.detectAndCompute(img_gray)
            keypoints_list.append(kps)
            descriptors_list.append(des)

        frame = Frame(
            timestamp=timestamp,
            images=images,
            images_gray=images_gray,
            keypoints=keypoints_list,
            descriptors=descriptors_list,
        )
        self.frames.append(frame)
        self.logger.info(f"[Initializer] Добавлен кадр (всего {len(self.frames)})")
        return frame
    
    def initialize(self, images, timestamp, depth=None) -> bool:

        if self.config.camera.is_mono:
            return self._initialize_mono(images[0], timestamp)

        elif self.config.camera.is_stereo:
            if images[1] is None:
                raise ValueError("Для стерео SLAM необходимо передать image_right.")
            return self._initialize_stereo(images[0], images[1], timestamp)

        elif self.config.camera.is_rgbd:
            if depth is None:
                raise ValueError("Для RGBD SLAM необходимо передать depth карту.")
            return self._initialize_rgbd(images[0], depth, timestamp)

        else:
            raise NotImplementedError(
                f"Неизвестный тип сенсора: {self.config.camera.sensor_type.name}"
            )
        
    def _initialize_mono(self, image, timestamp) -> bool:
        
        self.add_frame(images=[image], timestamp=timestamp)
        
        if len(self.frames) < 2:
            self.logger.info("[Initializer] Недостаточно кадров в очереди для инициализации.")
            return False
        
        f_cur = self.frames[-1]
        
        Kinv = self.camera.get_intrinsics_inv()

        for i, f_ref in enumerate(list(self.frames)[:-1]):

            self.logger.info(f"[Initializer] Новый кадр {i}")

            if not self._can_initialize(f_ref, f_cur):
                self.logger.info(f"[Initializer] Не прошёл _can_initialize")
                continue
            
            feature_result = self.tracking.feature_tracker.track(
                img1=f_cur.image_gray_left,
                img2=f_ref.image_gray_left,
                kps1=f_cur.keypoints_left,
                des1=f_cur.descriptors_left,
                kps2=f_ref.keypoints_left,
                des2=f_ref.descriptors_left,
                logger=self.logger,
                filtered_params=self.config.features.filtered_params
            )

            matches = feature_result.matches
            if len(matches) < self.min_inliers:
                self.logger.info(f"[Initializer] Недостаточно совпадений после нахождения точек и матчей: {len(matches)} / {self.min_inliers}")
                continue

            pts_cur = feature_result.kps1_matched
            pts_ref = feature_result.kps2_matched

            # self.viz.draw_matches(
            #     f_cur.image_gray_left,
            #     f_ref.image_gray_left,
            #     f_cur.keypoints_left,
            #     f_ref.keypoints_left,
            #     matches,
            #     window_name=f"Matches for Initialization {i} - {timestamp}",
            #     max_display=None
            # )
            
            pts_cur_n = normalize(Kinv, pts_cur)
            pts_ref_n = normalize(Kinv, pts_ref)

            R, t, inliers, mask_pose = estimate_motion_from_2d2d(
                pts_ref_n, pts_cur_n, logger=self.logger
            )
            
            if R is None or inliers < self.min_inliers:
                self.logger.info(f"[Initializer] Ошибка: инициализация не удалась. Инлайеров: {inliers}, лимит: {self.min_inliers}.")
                continue
            self.logger.info(f"[Initializer] Поза восстановлена. Инлайеров: {inliers}, лимит: {self.min_inliers}.")           
            
            R = R.T
            t = -R @ t
            
            tx, ty, tz = t.flatten()

            rpy = np.degrees(rpy_from_rotation_matrix(R, degrees=True))
            roll, pitch, yaw = rpy

            self.logger.info(
                f"[Initializer] ΔPose: "
                f"x={tx:.4f} m, y={ty:.4f} m, z={tz:.4f} m | "
                f"roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°"
            )
            
            f_ref.pose = np.eye(4)
            f_cur.set_pose(R, t)
            
            min_parallax_deg = self.config.tracking.min_parallax_deg
            
            parallax = compute_normalize_parallax(
                pts_ref_n=pts_ref_n,
                pts_cur_n=pts_cur_n,
                R=R
            )

            self.logger.info(f"[Initializer] parallax={parallax:.2f}° - лимит {min_parallax_deg / 2}°")

            if parallax < min_parallax_deg / 2:
                self.logger.info("[Initializer] Вырожденная сцена:малый параллакс.")
                continue

            # --- Триангуляция ---
            pts_3d, mask_triang = triangulate_normalized_points(f_ref.pose, f_cur.pose, pts_ref_n, pts_cur_n)

            pts_ref_n = pts_ref_n[mask_triang]
            pts_cur_n = pts_cur_n[mask_triang]
            pts_3d = pts_3d[mask_triang]
            
            if len(pts_3d) < self.min_inliers:
                self.logger.info(f"[Initializer] Недостаточно валидных 3D-точек после триангуляции: {len(pts_3d)}")
                continue

            # --- Фильтрация по параллаксу ---
            mask_parallax, ang = filter_by_parallax(pts_ref_n, pts_cur_n, R, min_parallax_deg=min_parallax_deg)

            pts_ref_n = pts_ref_n[mask_parallax]
            pts_cur_n = pts_cur_n[mask_parallax]
            pts_3d = pts_3d[mask_parallax]

            # Теперь всё синхронно, и можно безопасно использовать
            self.logger.info(f"[Initializer] Точек после фильтрации по параллаксу: {len(pts_3d)} / {len(mask_triang)} исходных")
            
            # pts_3d, t = normalize_depth_scale(pts_3d, t)
            
            # median_depth = np.median(pts_3d[:, 2])
            # ratio_depth_baseline = median_depth / (np.linalg.norm(t) + 1e-6)

            # if ratio_depth_baseline > 100 or ratio_depth_baseline < 1.0:
            #     self.logger.info(
            #         f"[Initializer] Неверное соотношение depth/baseline ({ratio_depth_baseline:.2f}), пропуск пары."
            #     )
            #     continue
            
            self.logger.info(
                f"[Initializer] Успешная пара {i} / {len(self.frames)-1}"
            )
            self._finalize_initialization(
                f_ref=f_ref,
                f_cur=f_cur,
                R=R,
                t=t,
                pts_3d=pts_3d,
                feature_result=feature_result,
            )
            return True

        self.num_failures += 1
        if self.num_failures % 5 == 0:
            self.min_inliers = max(30, self.min_inliers - 10)
            self.logger.warning(f"[Initializer] Снижаю порог min_matches={self.min_inliers}")
        return False
    
    def _initialize_stereo(self, image_left, image_right, timestamp) -> bool:
        pass

    def _initialize_rgbd(self, image_left, depth, timestamp) -> bool:
        pass
    
    def _finalize_initialization(
        self, 
        f_ref: Frame, 
        f_cur: Frame, 
        R, 
        t, 
        pts_3d: np.ndarray,
        feature_result: "FeatureTrackingResult" = None
    ):
        
        self.logger.info("[Initializer] Начало финального этапа инициализации")
        kf_ref = KeyFrame(
            pose=np.eye(4),
            keypoints=f_ref.keypoints_left,
            descriptors=f_ref.descriptors_left,
            timestamp=f_ref.timestamp,
        )

        kf_cur = KeyFrame(
            pose=poseRt(R, t),
            keypoints=f_cur.keypoints_left,
            descriptors=f_cur.descriptors_left,
            timestamp=f_cur.timestamp,
        )
        
        if self.config.debug:
            kf_ref.add_image(f_ref.image_left)
            kf_cur.add_image(f_cur.image_left)

        # self.viz.draw_keypoints(kf_ref.image, kf_ref.keypoints, window_name="After filtering")
        # self.viz.draw_keypoints(kf_cur.image, kf_cur.keypoints, window_name="After filtering")

        self.tracking.map.add_keyframe(kf_ref)
        self.tracking.map.add_keyframe(kf_cur)
        
        num_points_added = 0
        if feature_result is not None:

            img_ref = f_ref.image_left

            for i, (idx_cur, idx_ref) in enumerate(zip(feature_result.idxs1, feature_result.idxs2)):
                if i >= len(pts_3d):
                    break

                p3d = pts_3d[i]
                color = None
                if img_ref is not None:
                    u, v = np.round(feature_result.kps2_matched[i]).astype(int)
                    if 0 <= v < img_ref.shape[0] and 0 <= u < img_ref.shape[1]:
                        color = img_ref[v, u].astype(np.float32)

                mp = MapPoint(position=p3d, color=color)
                self.tracking.map.add_map_point(mp)

                kf_ref.add_point_match(mp, idx_ref)
                kf_cur.add_point_match(mp, idx_cur)
                mp.add_observation(kf_ref.id, idx_ref)
                mp.add_observation(kf_cur.id, idx_cur)

                num_points_added += 1
        else:
            for p in pts_3d:
                mp = MapPoint(position=p)
                self.tracking.map.add_map_point(mp)
                num_points_added += 1

        self.initialized = True
        
        # TODO разобраться зачем мне тут эти кадры
        self.tracking.last_frame = f_ref
        self.tracking.current_frame = f_cur
        self.tracking.reference_keyframe = kf_ref
        
        self.frames.clear()

        self.logger.info(
            f"[Initializer] Добавлено {num_points_added} 3D точек. "
            f"KeyFrames: {kf_ref.id}, {kf_cur.id}"
        )

        self._init_optimize()

        self.logger.info("[Initializer] Инициализация завершена успешно.\n")
        
    def _can_initialize(self, f_ref: Frame, f_cur: Frame) -> bool:
        # 1. Проверка расстояния между кадрами по timestamp
        dt = abs(f_cur.timestamp - f_ref.timestamp)
        if dt < 0.05:
            self.logger.info(f"[_can_initialize] Кадры слишком близко по времени ({dt:.3f}s)")
            return False

        # 2. Проверка количества фич
        if len(f_ref.keypoints_left) < self.min_inliers or len(f_cur.keypoints_left) < self.min_inliers:
            self.logger.info(f"[_can_initialize] Недостаточно фич для инициализации.")
            return False

        # 3. Проверка среднего распределения фич по изображению
        if not check_feature_coverage(f_ref.keypoints_left, f_cur.keypoints_left, f_ref.image_left.shape):
            self.logger.info(f"[_can_initialize] Фичи распределены неравномерно по кадру.")
            return False
        
        return True
    
    def _init_optimize(self):
        # TODO: реализовать оптимизацию начальной карты
        
        self.logger.info("[Initializer] Начали оптимизацию")
        
        self.logger.info("[Initializer] Закончили оптимизацию")
        pass
    
    