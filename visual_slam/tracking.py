import random
import time
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

import cv2
import numpy as np

from visual_slam.state import State
from visual_slam.initializer import Initializer
from visual_slam.map.map import Map
from visual_slam.map.map_point import MapPoint
from visual_slam.map.keyframe import KeyFrame
from visual_slam.utils.logging import get_logger
from visual_slam.frame import Frame
from visual_slam.utils.motion_estimation import (
    image_to_gray,
    triangulate_stereo_points,
    estimate_motion_from_2d3d
)
from visual_slam.utils.camera import backproject_3d
from visual_slam.utils.keypoints import filter_keypoints
from visual_slam.utils.matching import filter_matches

if TYPE_CHECKING:
    from visual_slam.slam import SLAM
    from visual_slam.config import Config
    


class Tracking:

    def __init__(
        self, 
        slam: "SLAM", 
        config: "Config",
        log_dir="logs"
    ):
        
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )
        
        self.slam = slam
        self.config = config

        self.initializer = Initializer(
            self, 
            config=config
        )
        
        self.state = State.NO_IMAGES_YET
        
        # История
        self.frame_id = 0
        self.last_frame: Optional[Frame] = None
        self.current_frame: Optional[Frame] = None
        self.reference_keyframe: Optional[KeyFrame] = None
        self.motion_model = np.eye(4)

    @property
    def state(self):
        return self.slam.state

    @state.setter
    def state(self, value):
        self.slam.state = value

    @property
    def feature_tracker(self):
        return self.slam.feature_tracker
    
    @property
    def map(self) -> Map:
        return self.slam.map
    
    @property
    def camera(self):
        return self.slam.camera
       
    @property
    def local_mapping(self):
        return self.slam.local_mapping

    # ---------------------------------------------------------
    # Основной цикл SLAM
    # ---------------------------------------------------------

    def track(
        self,
        images: List[np.ndarray],
        timestamp: float,
        depth: Optional[np.ndarray] = None
    ):
        """Главная функция трекинга, вызывается на каждый кадр."""
        if self.state == State.NO_IMAGES_YET:
            self._process_first_frame(images, timestamp)
            return self.state

        elif self.state in (State.NOT_INITIALIZED, State.INITIALIZING):
            self._try_initialize(images, timestamp, depth)
            return self.state

        elif self.state == State.OK:
            # self._track_ok(images, timestamp)
            return self.state

        elif self.state == State.LOST:
            self._relocalize(images, timestamp)
            return self.state
        
        else:
            self.logger.warning(f"[Tracking] Неизвестное состояние трекинга: {self.state}")
            return self.state

    # ---------------------------------------------------------
    # Стадии инициализации
    # ---------------------------------------------------------

    def _process_first_frame(self, images, timestamp):
        """Обработка самого первого кадра."""
        if self.config.camera.is_mono:
            self.initializer.add_frame(images, timestamp)
            self.logger.info("[Tracking] Первый кадр добавлен в очередь инициализации.")
        self.state = State.NOT_INITIALIZED
        self.logger.info("[Tracking] Переход в состояние NOT_INITIALIZED.")
        return self.state

    def _try_initialize(self, images, timestamp, depth=None):
        """Попытка инициализации карты."""
        self.state = State.INITIALIZING
        success = self.initializer.initialize(images, timestamp, depth)
        if success:
            self.state = State.OK
            self.logger.info("[Tracking] Инициализация успешна, переход в состояние OK.")
        else:
            self.state = State.NOT_INITIALIZED
            self.logger.warning("[Tracking] Инициализация не удалась, возвращение в состояние NOT_INITIALIZED.")
        return self.state
   
    # ---------------------------------------------------------
    # Основной цикл трекинга (OK)
    # ---------------------------------------------------------

    def _track_ok(
        self,
        images: List[np.ndarray],
        timestamp: float,
        depth: Optional[np.ndarray] = None,
    ):
        """Главный цикл трекинга после инициализации."""
        
        self.logger.info(
            f"[Tracking] Frame {self.frame_id}: трекинг активен."
        )
        
        if self.config.camera.is_mono:
            return self._track_mono(images, timestamp)
        elif self.config.camera.is_stereo:
            return self._track_stereo(images, timestamp)
        elif self.config.camera.is_rgbd:
            return self._track_rgbd(images, timestamp, depth)

    def debug_check_2d3d_projection(self, frame):
            """
            Визуальная проверка 2D–3D соответствий для текущего кадра.
            Зелёные точки — проекция 3D MapPoints.
            Красные точки — исходные keypoints.
            Линии показывают репроекционную ошибку.
            """
            import cv2
            import numpy as np

            if not frame.map_points or len(frame.map_points[0]) == 0:
                self.logger.warning("[Debug-2D3D] У кадра нет связанных MapPoints.")
                return

            img = frame.images_gray[0]
            if len(img.shape) == 2:
                img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_vis = img.copy()

            K = self.camera.K
            R = frame.pose[:3, :3]
            t = frame.pose[:3, 3].reshape(3, 1)

            for idx, mp in frame.map_points[0].items():
                if mp is None or not mp.is_valid():
                    continue

                # 3D → 2D проекция
                X = mp.position.reshape(3, 1)
                proj = K @ (R @ X + t)
                proj = proj[:2] / proj[2]

                u_proj, v_proj = int(proj[0]), int(proj[1])
                kp = frame.keypoints[0][idx]
                u_kp, v_kp = map(int, kp.pt)

                # Рисуем
                cv2.circle(img_vis, (u_kp, v_kp), 3, (0, 0, 255), -1)   # красный — keypoint
                cv2.circle(img_vis, (u_proj, v_proj), 3, (0, 255, 0), -1)  # зелёный — проекция
                cv2.line(img_vis, (u_kp, v_kp), (u_proj, v_proj), (255, 255, 0), 1)

            cv2.imshow("Debug 2D-3D Projection", img_vis)
            cv2.waitKey(0)
            cv2.destroyWindow("Debug 2D-3D Projection")

    def _track_mono(
        self,
        images: List[np.ndarray],
        timestamp: float,
    ):
        """Трекинг для монокамеры."""
        
        self.logger.info(f"[Tracking-MONO] Frame {self.frame_id}: трекинг монокамеры.")
        
        image = images[0]
        
        # Подготовка кадра
        self.current_frame = self._create_frame_mono(image, timestamp)
        
        self.logger.info(
            f"[Tracking-MONO] Frame {self.frame_id}: подготовлен кадр:"
            f"  Ключевых точек: {len(self.current_frame.keypoints[0])}."
            f"  Дескрипторов: {len(self.current_frame.descriptors[0])}."
        )
        
        # Предсказание позы (motion model)
        self._predict_pose()

        # Сопоставление с локальной картой
        match, pts_ref, pts_cur, pts_3d_pose = self._track_local_map_mono()

        # Оптимизация позы по PnP
        success, R, t, inliers = self._optimize_pose_mono(pts_cur, pts_3d_pose)
        if self.config.debug:
            self.logger.info("[Debug-2D3D] Проверка проекций 3D-точек на текущем кадре.")
            self.debug_check_2d3d_projection(self.current_frame)

        # Проверка качества трекинга
        if not self._is_tracking_good(
            sucscess_pnp=success,
            num_inliers=inliers,
            num_matches=len(match)
        ):
            self.logger.warning("[Tracking] Потеря трекинга — переход в LOST.")
            self.state = State.LOST
            return self.state

        # Проверка, нужен ли новый keyframe
        if self._need_new_keyframe():
            self._create_keyframe_mono()

        # Обновление состояния
        self._update_tracking_state()
        
        return self.state

    def _create_frame_mono(self, image: np.ndarray, timestamp: float) -> Frame:
        image_gray = image_to_gray(image)

        kps, des = self.feature_tracker.detectAndCompute(image_gray)
        if kps is None or len(kps) == 0:
            self.logger.warning("[Tracking-MONO] Не удалось извлечь фичи (MONO).")
            return None
        
        kps, des = filter_keypoints(
            kps,
            des,
            logger=self.logger,
            **self.config.features.filtered_params
        )

        frame = Frame(
            timestamp=timestamp,
            images=[image],
            images_gray=[image_gray],
            keypoints=[kps],
            descriptors=[des],
        )

        self.logger.info(f"[Tracking-MONO] Извлечено {len(kps)} фич.")
        return frame
    
    def _track_local_map_mono(self) -> Optional[dict]:
        
        def __debug_check_match(img_ref, img_cur, kps_ref, kps_cur, matches):
            """
            Интерактивная визуализация пар совпавших фич:
            - REF (предыдущий кадр) сверху, CUR (текущий) снизу
            - показывается одна пара за раз
            - навигация:
                [A] ← предыдущая пара
                [D] → следующая пара
                [Q] выход
            """
            if not matches:
                print("[DebugCheckMatch] Нет совпадений для отображения.")
                return

            img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
            img_cur = cv2.cvtColor(img_cur, cv2.COLOR_GRAY2BGR)
            h_ref, w_ref = img_ref.shape[:2]
            h_cur, w_cur = img_cur.shape[:2]
            H, W = h_ref + h_cur, max(w_ref, w_cur)

            idx = 0
            print(f"[DebugCheckMatch] Всего совпадений: {len(matches)}")
            print("Управление: [A] ← | [D] → | [Q] выход")

            while True:
                m = matches[idx]
                kp_ref = kps_ref[m.queryIdx]
                kp_cur = kps_cur[m.trainIdx]

                img_ref_vis = img_ref.copy()
                img_cur_vis = img_cur.copy()

                # Точки
                x1, y1 = map(int, kp_ref.pt)
                x2, y2 = map(int, kp_cur.pt)
                color = tuple(int(c) for c in np.random.randint(0, 255, 3))

                cv2.circle(img_ref_vis, (x1, y1), 6, color, -1)
                cv2.circle(img_cur_vis, (x2, y2), 6, color, -1)

                combined = np.zeros((H, W, 3), dtype=np.uint8)
                combined[:h_ref, :w_ref] = img_ref_vis
                combined[h_ref:h_ref+h_cur, :w_cur] = img_cur_vis

                # Соединяющая линия
                pt1, pt2 = (x1, y1), (x2, y2 + h_ref)
                cv2.line(combined, pt1, pt2, color, 2)

                cv2.putText(combined, f"Match {idx+1}/{len(matches)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "REF", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, "CUR", (20, h_ref + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("DebugCheckMatch", combined)
                key = cv2.waitKey(0) & 0xFF

                if key in [ord('q'), ord('Q')]:
                    cv2.destroyWindow("DebugCheckMatch")
                    break
                elif key in [ord('d'), ord('D')]:
                    idx = (idx + 1) % len(matches)
                elif key in [ord('a'), ord('A')]:
                    idx = (idx - 1) % len(matches)
        
        f_cur = self.current_frame
        f_ref = self.last_frame        
        kf_ref = self.reference_keyframe

        des_cur = f_cur.descriptors[0]
        des_ref = f_ref.descriptors[0]
        
        kps_ref = f_ref.keypoints[0]
        kps_cur = f_cur.keypoints[0]
        
        # Сопоставление фич между кадрами
        matches = self.feature_tracker.match(des_ref, des_cur)  
        self.logger.info(f"[Tracking-MONO] Найдено {len(matches)} исходных совпадений между кадрами.")      
        matches = filter_matches(
            matches,
            kps1=kps_ref,
            kps2=kps_cur,
            logger=self.logger,
            **self.config.features.filtered_params
        )        
        self.logger.info(f"[Tracking-MONO] После фильтрации осталось {len(matches)} совпадений.")
        if len(matches) < self.config.tracking.min_inliers:
            self.logger.warning(f"[Tracking-MONO] Недостаточно матчей ({len(matches)}), min_inliers={self.config.tracking.min_inliers}.")
            
        # __debug_check_match(
        #     f_ref.images_gray[0],
        #     f_cur.images_gray[0],
        #     kps_ref,
        #     kps_cur,
        #     matches
        # )
        
        pts_cur = []
        pts_3d_pos = []
        
        f_cur.map_points.append({})

        for m in matches:
            mp: "MapPoint" = kf_ref.get_point_match(m.queryIdx)
            if mp is not None and mp.is_valid():
                pts_3d_pos.append(mp.position)
                pts_cur.append(kps_cur[m.trainIdx].pt)
                f_cur.map_points[0][m.trainIdx] = mp
        
        num_valid = len(pts_3d_pos)
        self.logger.info(f"[Tracking-MONO] Из {len(matches)} матчей найдено {num_valid} совпадений с прошлым ключевым кадром.")

        f_cur.num_visible_mappoints = num_valid

        if num_valid < 20:
            self.logger.warning(f"[Tracking-MONO] Мало валидных 3D-точек ({num_valid}).")

        pts_cur = np.array(pts_cur, dtype=np.float32)
        pts_ref = np.array([kps_ref[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_3d_pose = np.array(pts_3d_pos, dtype=np.float32)  
        
        self.logger.info(f"[Tracking-MONO] Подготовлено {len(pts_3d_pose)} 3D-точек для PnP.")
        
        return matches, pts_ref, pts_cur, pts_3d_pose
    
    def _optimize_pose_mono(self, pts_cur, pts_3d_pose):
        if pts_3d_pose is None or pts_cur is None or len(pts_3d_pose) < 4:
            self.logger.warning("[Tracking-MONO] Недостаточно 3D-точек для PnP.")
            return False, None, None, 0
        
        self.logger.info(
            f"[Tracking-MONO] PnP: старт оптимизации. "
            f"  Точек: {len(pts_3d_pose)}, "
        )

        K = self.camera.K
        dist = self.camera.dist
        
        self.logger.debug(f"[Tracking-MONO] Camera K:\n{K}")
        self.logger.debug(f"[Tracking-MONO] Distortion coeffs: {dist}")

        R, t, success, inliers_mask = estimate_motion_from_2d3d(
            pts3d=pts_3d_pose,
            pts2d=pts_cur,
            K=K,
            dist=dist,
            use_ransac=True,
        )

        if not success or R is None or t is None:
            self.logger.warning("[Tracking-MONO] PnP не удалось.")
            return False, None, None, 0
        
        det_R = np.linalg.det(R)
        t_norm = np.linalg.norm(t)
        num_inliers = int(inliers_mask.shape[0]) if inliers_mask is not None else 0

        self.logger.info(
            f"[Tracking-MONO] PnP успешно выполнен: "
            f"inliers={num_inliers}, det(R)={det_R:.6f}, |t|={t_norm:.4f}"
        )

        self.logger.debug(f"[Tracking-MONO] Матрица R:\n{R}")
        self.logger.debug(f"[Tracking-MONO] Вектор t:\n{t.reshape(-1)}")

        # Обновляем позу текущего кадра
        self.current_frame.set_pose(R, t)

        self.logger.info(
            f"[Tracking-MONO] Обновлена поза текущего кадра. "
            f"t=({t[0][0]:.4f}, {t[1][0]:.4f}, {t[2][0]:.4f})"
        )

        if inliers_mask is not None:
            inlier_ratio = num_inliers / len(pts_cur)
            self.logger.info(
                f"[Tracking-MONO] PnP инлайеры: {num_inliers}/{len(pts_cur)} "
                f"({inlier_ratio*100:.2f}%)."
            )
        else:
            self.logger.warning("[Tracking-MONO] PnP: маска инлайеров отсутствует.")

        return True, R, t, num_inliers
    
    def _create_keyframe_mono(self):

        f_cur = self.current_frame

        # Создаём KeyFrame
        kf = KeyFrame(
            pose=f_cur.pose,
            keypoints=f_cur.keypoints[0],
            descriptors=f_cur.descriptors[0],
            timestamp=f_cur.timestamp,
        )
        if self.config.debug:
            kf.add_image(f_cur.images[0])

        self.logger.info(f"[Create KF] Создан новый KeyFrame #{kf.id} из Frame {self.frame_id}.")

        # Привязка видимых 3D-точек (MapPoints)
        num_associated = 0
        mp_dict = f_cur.map_points[0]

        # Проверяем, что у кадра есть карта соответствий map_points
        for idx_kp, mp in mp_dict.items():
            if mp is not None and mp.is_valid():
                # создаём постоянную связь между KeyFrame и MapPoint
                kf.add_point_match(mp, idx_kp)
                mp.add_observation(kf.id, idx_kp)
                num_associated += 1

                # добавляем MapPoint в карту, если он ещё не там
                if mp not in self.map.get_points():
                    self.map.add_map_point(mp)
                    self.logger.info("[Create KF] Добавляем MapPoint в карту, так как ещё её нет там")
                    
        self.logger.info(f"[Create KF] Привязано {num_associated} существующих MapPoints.")

        self.map.add_keyframe(kf)

        self.logger.info(
            f"[KeyFrame] KeyFrame #{kf.id} добавлен в карту. "
            f"  Всего KeyFrames: {self.map.num_keyframes()}, "
            f"MapPoints: {self.map.num_points()}."
        )

        return kf
        
    def _track_stereo(
        self,
        images: List[np.ndarray],
        timestamp: float,
    ):
        if len(images) < 2:
            self.logger.error("[Tracking] Для стерео требуется два изображения.")
            return None

        image_left, image_right = images[:2]
        
        # Подготовка кадра
        frame = self._create_frame_stereo(image_left, image_right, timestamp)
    
    def _create_frame_stereo(
        self, 
        image_left: np.ndarray, 
        image_right: np.ndarray,
        timestamp: float
    ) -> Frame:
        left_gray = image_to_gray(image_left)
        right_gray = image_to_gray(image_right)

        kps_left, des_left = self.feature_tracker.detectAndCompute(left_gray)
        kps_right, des_right = self.feature_tracker.detectAndCompute(right_gray)

        if not kps_left or not kps_right:
            self.logger.warning("[Tracking] Не удалось извлечь фичи (STEREO).")
            return None

        matches = self.feature_tracker.match(des_left, des_right)
        if len(matches) == 0:
            self.logger.warning("[Tracking] Нет совпадений между левым и правым изображениями (STEREO).")
            return None

        pts_left = np.array([kps_left[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_right = np.array([kps_right[m.trainIdx].pt for m in matches], dtype=np.float32)

        K = self.camera.K
        baseline = getattr(self.camera, "baseline", 0.1)
        pts_3d = triangulate_stereo_points(pts_left, pts_right, K, baseline)

        frame = Frame(
            timestamp=timestamp,
            images=[image_left, image_right],
            images_gray=[left_gray, right_gray],
            keypoints=[kps_left, kps_right],
            descriptors=[des_left, des_right],
            points_3d=pts_3d,
        )

        self.logger.info(
            f"[Tracking] STEREO: {len(kps_left)} фич (левое), {len(matches)} совпадений, {len(pts_3d)} 3D точек."
        )
        return frame
    
    def _track_rgbd(
        self,
        images: List[np.ndarray],
        timestamp: float,
        depth: np.ndarray,
    ):
        """Трекинг для RGB-D камеры."""
        if depth is None:
            self.logger.error("[Tracking] Для RGB-D требуется depth map.")
            return None

        image_rgb = images[0]

        # Подготовка кадра
        frame = self._create_frame_rgbd(image_rgb, timestamp, depth)
    
    def _create_frame_rgbd(self, image: np.ndarray, timestamp: float, depth: np.ndarray) -> Frame:
        
        image_gray = image_to_gray(image)

        kps, des = self.feature_tracker.detectAndCompute(image_gray)
        if kps is None or len(kps) == 0:
            self.logger.warning("[Tracking] Не удалось извлечь фичи (RGBD).")
            return None

        uv = np.array([kp.pt for kp in kps], dtype=np.float32)
        h, w = depth.shape
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)

        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[valid_mask], v[valid_mask]
        z = depth[v, u]

        valid_depth = z > 1e-3
        if np.sum(valid_depth) < 10:
            self.logger.warning("[Tracking] Недостаточно валидных глубин (RGBD).")
            return None

        uv_valid = np.stack((u[valid_depth], v[valid_depth]), axis=1)
        z_valid = z[valid_depth]

        K = self.camera.K
        pts_3d = backproject_3d(uv_valid, z_valid, K)

        frame = Frame(
            timestamp=timestamp,
            images=[image],
            images_gray=[image_gray],
            keypoints=[kps],
            descriptors=[des],
            points_3d=pts_3d,
            depth=depth,
        )

        self.logger.info(f"[Tracking] RGB-D: извлечено {len(kps)} фич, {len(pts_3d)} 3D точек.")
        return frame

    def _predict_pose(self):
        """Априорная оценка движения текущего кадра (motion model)."""
        if self.last_frame is None:
            self.current_frame.pose = np.eye(4)
            self.logger.info("[Tracking] Нет предыдущего кадра — поза = identity.")
            return

        self.current_frame.pose = self.last_frame.pose @ self.motion_model
        self.logger.info("[Tracking] Преобразование через motion model.")

    def _is_tracking_good(
        self,
        sucscess_pnp: bool,
        num_inliers: int,
        num_matches: int,
        reprojection_error: Optional[float] = None
    ) -> bool:

        cfg = self.config.tracking
        min_inliers = getattr(cfg, "min_inliers", 8)
        min_inlier_ratio = getattr(cfg, "min_inlier_ratio", 0.25)
        max_reproj_error = getattr(cfg, "max_reprojection_error", 5.0)
        warmup_frames = getattr(cfg, "warmup_frames", 30)
        
        if self.frame_id < warmup_frames:
            scale = 0.5 + 0.5 * (self.frame_id / warmup_frames)
            min_inliers = int(min_inliers * scale)
            min_inlier_ratio = min_inlier_ratio * 0.5
            self.logger.debug(
                f"[Is Tracking good] Адаптивный режим: frame={self.frame_id}, "
                f"scale={scale:.2f}, min_inliers={min_inliers}, min_ratio={min_inlier_ratio:.2f}"
            )
        
        if not sucscess_pnp:
            self.logger.warning("[Is Tracking good] PnP неуспешен.")
            return False
        
        self.logger.info("[Is Tracking good] Проверка на PnP пройдена.")

        # Проверка количества инлайеров
        if num_inliers < min_inliers:
            self.logger.warning(
                f"[Is Tracking good] Мало инлайеров ({num_inliers} < {min_inliers})"
            )
            return False
        
        self.logger.info("[Is Tracking good] Проверка количества инлайеров пройдена.")

        # Проверка отношения инлайеров к матчам
        if num_matches > 0:
            inlier_ratio = num_inliers / num_matches
            if inlier_ratio < min_inlier_ratio:
                self.logger.warning(
                    f"[Is Tracking good] Низкое отношение инлайеров ({inlier_ratio:.2f} < {min_inlier_ratio})"
                )
                return False
        
        self.logger.info("[Is Tracking good] Проверка отношения инлайеров к матчам пройдена.")

        # Проверка ошибки репроекции (если передана)
        if reprojection_error is not None and reprojection_error > max_reproj_error:
            self.logger.warning(
                f"[Is Tracking good] Большая ошибка репроекции ({reprojection_error:.2f} > {max_reproj_error})"
            )
            return False
        
        self.logger.info("[Is Tracking good] Проверка ошибки репроекции пройдена.")

        self.logger.info(
            f"[Is Tracking good] Трекинг устойчив: {num_inliers} инлайеров ({inlier_ratio:.2f})."
        )
        return True

    def _need_new_keyframe(self) -> bool:

        cfg = self.config.tracking

        # Проверка интервала по кадрам
        keyframe_interval = getattr(cfg, "keyframe_interval", 5)
        if self.frame_id - self.reference_keyframe.id < keyframe_interval:
            self.logger.debug(f"[KeyFrame] Интервал < {keyframe_interval} — пока не создаём.")
            return False

        # Проверка количества инлайеров
        # last_inliers = getattr(self, "last_num_inliers", 0)
        # cur_inliers = getattr(self.current_frame, "num_inliers", 0)
        # low_inlier_ratio = getattr(cfg, "low_inlier_ratio", 0.75)

        # if last_inliers > 0 and cur_inliers / last_inliers < low_inlier_ratio:
        #     self.logger.info(
        #         f"[KeyFrame] Качество трека снизилось "
        #         f"({cur_inliers}/{last_inliers}) — возможен новый keyframe."
        #     )
        #     return True

        # Проверка смещения камеры
        if self.last_frame is not None:
            T_ref = self.last_frame.pose
            T_cur = self.current_frame.pose
            delta_T = np.linalg.inv(T_ref) @ T_cur
            
            trans = np.linalg.norm(delta_T[:3, 3])
            rot_angle = np.degrees(
                np.arccos(np.clip((np.trace(delta_T[:3, :3]) - 1) / 2.0, -1, 1))
            )

            max_trans = getattr(cfg, "max_translation_for_kf", 0.5)
            max_rot = getattr(cfg, "max_rotation_for_kf", 10.0)

            if trans > max_trans or rot_angle > max_rot:
                self.logger.info(
                    f"[KeyFrame] Значительное движение "
                    f"(Δt={trans:.3f} m, ΔR={rot_angle:.2f}°) — создаём новый keyframe."
                )
                return True

        # Проверка числа видимых картографических точек
        visible_points = getattr(self.current_frame, "num_visible_mappoints", 0)
        if visible_points < getattr(cfg, "min_visible_points_for_kf", 25):
            self.logger.info(
                f"[KeyFrame] Мало видимых точек ({visible_points}) — добавляем новый keyframe."
            )
            return True

        return False

    def _create_keyframe(self):

        f_cur = self.current_frame

        # Создаём KeyFrame
        kf = KeyFrame(
            pose=f_cur.pose,
            keypoints=f_cur.keypoints,
            descriptors=f_cur.descriptors,
            timestamp=f_cur.timestamp,
        )
        if self.config.debug:
            kf.add_image(f_cur.images[0])
        self.map.add_keyframe(kf)

        self.logger.info(f"[KeyFrame] Создан новый KeyFrame #{kf.id} из Frame {self.frame_id}.")

        # Привязка видимых 3D-точек (MapPoints)
        num_associated = 0

        # Проверяем, что у кадра есть карта соответствий map_points
        if f_cur.map_points is not None and isinstance(f_cur.map_points, dict):
            for idx_kp, mp in f_cur.map_points.items():
                if mp is not None and not mp.is_bad:
                    kf.add_point_match(mp, idx_kp)
                    mp.add_observation(kf.id, idx_kp)
                    num_associated += 1

        self.logger.info(f"[KeyFrame] Привязано {num_associated} существующих MapPoints.")

        # Добавляем новые точки (если у кадра есть 3D-точки)
        num_new_points = 0
        if f_cur.points_3d is not None and len(f_cur.points_3d) > 0:
            for p3d in f_cur.points_3d:
                if np.isfinite(p3d).all():
                    mp = MapPoint(position=p3d)
                    mp.add_observation(kf.id, -1)
                    self.map.add_map_point(mp)
                    num_new_points += 1

        self.logger.info(f"[KeyFrame] Добавлено {num_new_points} новых MapPoints (из кадра).")

        self.logger.info(
            f"[KeyFrame] KeyFrame #{kf.id} успешно добавлен в карту. "
        )

        return kf

    def _update_tracking_state(self):
        """Обновление переменных после успешного трекинга."""
        if self.last_frame is not None and np.isfinite(self.last_frame.pose).all() and np.isfinite(self.current_frame.pose).all():
            try:
                self.motion_model = np.linalg.inv(self.last_frame.pose) @ self.current_frame.pose
                self.logger.info("[Update var] Motion model обновлена.")
            except np.linalg.LinAlgError:
                self.motion_model = np.eye(4)
                self.logger.warning("[Update var] Ошибка инверсии позы — сброшена motion model.")
        else:
            self.motion_model = np.eye(4)
            self.logger.info("[Update var] Нет предыдущего кадра — motion model = identity.")
            
        self.last_frame = self.current_frame.copy()        
        self.reference_keyframe = self.map.get_keyframes()[-1]
        self.frame_id += 1
        self.logger.info(
            f"[Tracking] Обновление состояния завершено. "
            f"Текущий KF: #{self.reference_keyframe.id}, "
            f"ΔT_norm={np.linalg.norm(self.motion_model[:3, 3]):.4f}."
        )
        return None

    # ---------------------------------------------------------
    # Восстановление (relocalization)
    # ---------------------------------------------------------

    def _relocalize(self, image, timestamp):
        """Попытка восстановить трекинг после потери."""
        # TODO: сопоставление с базой keyframe, PnP
        self.logger.warning("[Tracking] Попытка релокализации.")
        return None

    # ---------------------------------------------------------
    # Сброс
    # ---------------------------------------------------------

    def reset(self):
        """Полный сброс трекинга и инициализатора."""
        self.logger.info("[Tracking] Сброс состояния трекинга.")
        self.state = State.NO_IMAGES_YET
        self.frame_id = 0
        self.last_frame = None
        self.current_frame = None
        self.reference_keyframe = None
        self.motion_model = np.eye(4)
        self.initializer.reset()
