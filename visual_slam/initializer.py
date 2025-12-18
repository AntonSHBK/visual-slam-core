from typing import List

import numpy as np

from visual_slam import camera
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint
from visual_slam.utils.logging import get_logger
from visual_slam.utils.motion_estimation import (
    estimate_motion_from_2d2d,
    triangulate_points,
    compute_normalize_parallax,
    filter_by_parallax,
    check_feature_coverage,
    image_to_gray,
    filter_points_by_depth
)

from visual_slam.utils.geometry import normalize
from visual_slam.viz.feature_viz import FeatureVisualizer
from visual_slam.map.frame import Frame

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
        self.map = self.tracking.map
        
        self.initialized: bool = False
        self.num_failures = 0
        
        self.viz = FeatureVisualizer(window_name_prefix="Initializer")

        self.logger.info("\n\n[Initializer] Инициализация ====================>\n")

    def reset(self):
        self.initialized = False
        self.min_inliers = self.config.tracking.min_inliers
        self.map.reset()
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
            camera=self.camera,
            images=images,
            images_gray=images_gray,
            keypoints=keypoints_list,
            descriptors=descriptors_list,
        )
        self.map.add_frame(frame=frame)
        self.logger.info(f"[Initializer] Добавлен кадр (всего {len(self.map.get_frames())})")
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
        f_cur = self.add_frame(images=[image], timestamp=timestamp)
        
        if len(self.map.get_frames()) < 2:
            self.logger.info("[Initializer] Недостаточно кадров в очереди для инициализации.")
            return False
        
        Kinv = self.camera.get_intrinsics_inv()

        for i, f_ref in enumerate(self.map.get_frames()[:-1]):

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
            
            # self.viz.draw_keypoints(
            #     f_cur.image_left,
            #     f_cur.keypoints_left             
            # )
            
            # self.viz.draw_keypoints(
            #     f_ref.image_left,
            #     f_ref.keypoints_left             
            # )
            
            # self.viz.draw_matches(
            #     f_cur.image_gray_left,
            #     f_ref.image_gray_left,
            #     f_cur.keypoints_left,
            #     f_ref.keypoints_left,
            #     matches,
            #     window_name=f"Matches for Initialization {i} - {timestamp}",
            #     max_display=None
            # )
            
            pts_cur_n = normalize(Kinv, feature_result.kps1_matched)
            pts_ref_n = normalize(Kinv, feature_result.kps2_matched)

            R_rel, t_rel, inliers, mask_pose = estimate_motion_from_2d2d(
                pts_ref=pts_ref_n, 
                pts_cur=pts_cur_n, 
            )
            
            T_ref2cur = np.eye(4)
            T_ref2cur[:3, :3] = R_rel
            T_ref2cur[:3, 3] = t_rel.reshape(3)
            T_w2c_ref = f_ref.T_w2c
            T_w2c_cur = T_w2c_ref @ T_ref2cur
            f_cur.update_pose(T_w2c_cur)           
            
            if self.config.debug:            
                R_w2c = f_cur.R_w2c      # rotation: world → camera
                t_w2c = f_cur.t_w2c      # translation: world → camera

                R_c2w = f_cur.R_c2w      # rotation: camera → world
                t_c2w = f_cur.t_c2w      # translation: camera → world

                # roll_world, pitch_world, yaw_world = rpy_from_rotation_matrix(R_c2w, degrees=True)
                # roll_local, pitch_local, yaw_local = rpy_from_rotation_matrix(R_w2c, degrees=True)
                
                self.logger.info(
                    "[Initializer] Поза камеры (global frame):\n"
                    f"  t_c2w = ({t_c2w[0]:.4f}, {t_c2w[1]:.4f}, {t_c2w[2]:.4f})\n"
                    # f"  R_global (RPY deg) = "
                    # f"({roll_world:.2f}, {pitch_world:.2f}, {yaw_world:.2f})"
                )

                self.logger.info(
                    "[Initializer] Преобразование мира в локальные координаты камеры:\n"
                    f"  t_w2c = ({t_w2c[0]:.4f}, {t_w2c[1]:.4f}, {t_w2c[2]:.4f})\n"
                    # f"  R_w2c (RPY deg) = "
                    # f"({roll_local:.2f}, {pitch_local:.2f}, {yaw_local:.2f})"
                )      
            
            if R_rel is None or inliers < self.min_inliers:
                self.logger.info(f"[Initializer] Ошибка: инициализация не удалась. Инлайеров: {inliers}, лимит: {self.min_inliers}.")
                continue
            self.logger.info(f"[Initializer] Поза восстановлена. Инлайеров: {inliers}, лимит: {self.min_inliers}.")                   
                       
            min_parallax_deg = self.config.tracking.min_parallax_deg
            
            parallax = compute_normalize_parallax(
                pts_ref_n=pts_ref_n,
                pts_cur_n=pts_cur_n,
                pose_ref=f_ref.T_w2c,
                pose_cur=f_cur.T_w2c,
            )

            self.logger.info(f"[Initializer] parallax={parallax:.2f}° - лимит {min_parallax_deg / 2}°")

            if parallax < min_parallax_deg / 2:
                self.logger.info("[Initializer] Вырожденная сцена:малый параллакс.")
                continue           

            pts_3d, mask_triang = triangulate_points(
                T_w2c_ref=f_ref.T_w2c,
                T_w2c_cur=f_cur.T_w2c,
                pts_ref_n=pts_ref_n,
                pts_cur_n=pts_cur_n
            )
            
            if len(pts_3d) < self.min_inliers:
                self.logger.info(f"[Initializer] Недостаточно валидных 3D-точек после триангуляции: {len(pts_3d)}")
                continue
            
            pts_ref_n = pts_ref_n[mask_triang]
            pts_cur_n = pts_cur_n[mask_triang]
            pts_3d = pts_3d[mask_triang]
            feature_result = feature_result.filter_by_mask(mask_triang)
            
            self.logger.info(f"[Initializer] Точек после триангуляции: {len(pts_3d)} / {len(mask_triang)} исходных")
            
            min_depth = self.config.initialization.min_depth
            max_depth = self.config.initialization.max_depth
            
            pts_3d, mask_depth = filter_points_by_depth(
                pts_3d,
                T_w2c_ref,
                T_w2c_cur,
                min_depth=min_depth,
                max_depth=50.0
            )

            pts_ref_n = pts_ref_n[mask_depth]
            pts_cur_n = pts_cur_n[mask_depth]
            feature_result = feature_result.filter_by_mask(mask_depth)  
            
            self.logger.info(f"[Initializer] Точек после фильтрации по глубине: {len(pts_3d)} / {len(mask_triang)} исходных")         

            mask_parallax, ang = filter_by_parallax(
                pts_ref_n=pts_ref_n,
                pts_cur_n=pts_cur_n,
                pose_ref=f_ref.T_w2c,
                pose_cur=f_cur.T_w2c,
                min_parallax_deg=min_parallax_deg
            )

            pts_ref_n = pts_ref_n[mask_parallax]
            pts_cur_n = pts_cur_n[mask_parallax]
            pts_3d = pts_3d[mask_parallax]
            feature_result = feature_result.filter_by_mask(mask_parallax)

            self.logger.info(f"[Initializer] Точек после фильтрации по параллаксу: {len(pts_3d)} / {len(mask_triang)} исходных")           
            
            self.logger.info(
                f"[Initializer] Успешная пара {i} / {len(self.map.get_frames())-1}"
            )
            self._finalize_initialization(
                f_ref=f_ref,
                f_cur=f_cur,
                points_3d_list=[pts_3d],
                feature_results=[feature_result],
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
        points_3d_list: list[np.ndarray],
        feature_results: list["FeatureTrackingResult"] = None
    ):
        self.logger.info("[Initializer] Начало финального этапа инициализации")
        
        kf_ref = KeyFrame.from_frame(f_ref)
        kf_cur = KeyFrame.from_frame(f_cur)

        self.tracking.map.add_keyframe(kf_ref)
        self.tracking.map.add_keyframe(kf_cur)

        num_points_added = 0
        
        feature_results = feature_results or []
        num_cams = len(points_3d_list)
        
        for cam_idx in range(num_cams):
            pts_3d = points_3d_list[cam_idx]
            feat_res = feature_results[cam_idx] if cam_idx < len(feature_results) else None

            if pts_3d is None or len(pts_3d) == 0:
                self.logger.warning(f"[Initializer] Камера {cam_idx}: нет 3D-точек для добавления.")
                continue

            img_ref = f_ref.get_image(cam_idx)
            kps_ref = f_ref.get_keypoints(cam_idx)

            if feat_res is None:
                for p3d in pts_3d:
                    mp = MapPoint(position=p3d)
                    self.tracking.map.add_map_point(mp)
                    num_points_added += 1
                continue

            for i, p3d in enumerate(pts_3d):
                if i >= len(feat_res.idxs1) or i >= len(feat_res.idxs2):
                    mp = MapPoint(position=p3d)
                    self.tracking.map.add_map_point(mp)
                    num_points_added += 1
                    continue

                idx_ref = feat_res.idxs2[i]
                idx_cur = feat_res.idxs1[i]

                color = None
                if img_ref is not None and len(kps_ref) > idx_ref:
                    u, v = np.round(kps_ref[idx_ref].pt).astype(int)
                    if 0 <= v < img_ref.shape[0] and 0 <= u < img_ref.shape[1]:
                        color = img_ref[v, u].astype(np.float32)

                mp = MapPoint(position=p3d, color=color)
                self.tracking.map.add_map_point(mp)

                kf_ref.add_map_point(
                    cam_id=cam_idx,
                    kp_idx=idx_ref,
                    map_point=mp
                )
                kf_cur.add_map_point(
                    cam_id=cam_idx,
                    kp_idx=idx_cur, 
                    map_point=mp
                )

                num_points_added += 1

        self.logger.info(
            f"[Initializer] Добавлено {num_points_added} 3D точек. "
            f"KeyFrames: {kf_ref.keyframe_id}, {kf_cur.keyframe_id}"
        )

        mean_err_before_optimize = self.tracking.map.compute_mean_reprojection_error(
            camera=self.camera,
            keyframes=[kf_cur]
        )
        self.logger.info(f"[Initializer] Средняя ошибка до оптимизации: {mean_err_before_optimize:.3f}px")
        
        optimized = self._optimize()

        mean_err_after_optimize = self.tracking.map.compute_mean_reprojection_error(
            camera=self.camera,
            keyframes=[kf_cur]
        )
        self.logger.info(f"[Initializer] Средняя ошибка после оптимизации: {mean_err_after_optimize:.3f}px")

        # TODO надо разобраться с масштабом карты, возможно вынести в map
        # self._normalize_map_scale()
        
        # mean_err_after_normalize = self.tracking.map.compute_mean_reprojection_error(
        #     camera=self.camera,
        #     keyframes=[kf_cur]
        # )
        # self.logger.info(f"[Initializer] Средняя ошибка после нормализации: {mean_err_after_normalize:.3f}px")
        
        self.initialized = True        
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
    
    def _normalize_map_scale(
        self,
        min_ratio: float = 1.0,
        max_ratio: float = 100.0
    ):
        """
        Нормализует масштаб карты после инициализации.
        Масштабируется так, чтобы медианная глубина 3D-точек ≈ 1.0.
        """
        keyframes = self.tracking.map.get_keyframes()
        points = self.tracking.map.get_points()

        if len(keyframes) < 2 or len(points) == 0:
            self.logger.warning("[Initializer] Недостаточно данных для нормализации карты.")
            return

        kf_ref = min(keyframes, key=lambda kf: kf.timestamp)
        kf_cur = max(keyframes, key=lambda kf: kf.timestamp)

        positions = np.array([p.position for p in points if p.is_valid()])
        if len(positions) == 0:
            self.logger.warning("[Initializer] Нет валидных 3D-точек для нормализации.")
            return

        median_depth = np.median(positions[:, 2])

        center_ref = kf_ref.t_c2w
        center_cur = kf_cur.t_c2w
        baseline = np.linalg.norm(center_cur - center_ref)

        ratio_depth_baseline = median_depth / (baseline + 1e-6)

        self.logger.info(
            f"[Initializer] До нормализации: "
            f"median_depth={median_depth:.3f}, baseline={baseline:.3f}, "
            f"ratio={ratio_depth_baseline:.2f}"
        )

        if ratio_depth_baseline > max_ratio or ratio_depth_baseline < min_ratio:
            self.logger.warning(
                f"[Initializer] Некорректное соотношение depth/baseline ({ratio_depth_baseline:.2f}). "
                f"Масштаб может быть неверным."
            )

        if median_depth > 1e-6:
            scale = 1.0 / median_depth
            
            self.logger.info(
                f"[Initializer] Масштаб применён: {scale:.3f}. "
            )

            for mp in points:
                mp.update_position(mp.position * scale)

            for kf in keyframes:
                kf.update_translation(kf.pose.t * scale)

            new_baseline = np.linalg.norm(
                kf_cur.pose.inverse().t - kf_ref.pose.inverse().t
            )
            
            ratio_depth_baseline = 1 / (new_baseline + 1e-6)

            self.logger.info(
                f"[Initializer] После нормализации: "
                f"median_depth=1.0, baseline={new_baseline:.3f}, "
                f"ratio={ratio_depth_baseline:.2f}"
            )
        else:
            self.logger.warning("[Initializer] Медианная глубина близка к нулю, нормализация пропущена.")
    
    def _optimize(self):
        self.logger.info("[Initializer] Начали оптимизацию")
        optimized = self.tracking.map.optimize_initial(optimizer=self.tracking.optimizer)
        self.logger.info("[Initializer] Закончили оптимизацию")
        return optimized
    
    