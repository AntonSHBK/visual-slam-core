from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from weakref import ref

from anyio import key
import cv2
import numpy as np
from sympy import li

from visual_slam.state import State
from visual_slam.initializer import Initializer
from visual_slam.map.map import Map
from visual_slam.map.pose import Pose
from visual_slam.map.map_point import MapPoint
from visual_slam.map.keyframe import KeyFrame
from visual_slam.optimization.torch_optimizer import TorchOptimizer
from visual_slam.utils.logging import get_logger
from visual_slam.map.frame import Frame
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
        
        self.logger.info(f"{'*'*80}")
        
        self.slam = slam
        self.config = config

        self.initializer = Initializer(
            self, 
            config=config
        )
        
        self.optimizer = TorchOptimizer(config)
        self.frames = self.map._frames
        
        self.state = State.NO_IMAGES_YET
        
        # История
        self.last_frame: Frame = None
        self.current_frame: Frame = None
        self.reference_keyframe: KeyFrame = None
        self.motion_model = Pose()
        
    def reset(self):
        self.logger.info("[Tracking] Сброс состояния трекинга.")
        self.state = State.NO_IMAGES_YET
        self.last_frame = None
        self.current_frame = None
        self.reference_keyframe = None
        self.motion_model = Pose()
        self.initializer.reset()

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
    def local_mapping(self):
        return self.slam.local_mapping
    
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
            self._track_ok(images, timestamp)
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
            f"[Tracking] =========== Трекинг кадра при состоянии OK. Timestamp = {timestamp}"
        )
        
        if self.config.camera.is_mono:
            return self._track_mono(images, timestamp)
        elif self.config.camera.is_stereo:
            return self._track_stereo(images, timestamp)
        elif self.config.camera.is_rgbd:
            return self._track_rgbd(images, timestamp, depth)

    def _track_mono(
        self,
        images: List[np.ndarray],
        timestamp: float,
    ):
        self.logger.info("[Tracking-MONO] Трекинг моно-камеры.")
        
        image = images[0]
        
        self.reference_keyframe = self.map.get_last_keyframe()
        self.last_frame = self.map.get_last_frame()
        self.current_frame = self._create_frame_mono(image, timestamp)
        
        self.logger.info(f'[Tracking-MONO] Reference KeyFrame ID: {self.reference_keyframe.id if self.reference_keyframe else "None"}.')
        
        self.logger.info(
            f"[Tracking-MONO] Frame {self.current_frame.id}: подготовлен кадр:"
            f" Ключевых точек: {len(self.current_frame.keypoints[0])}."
            f" Дескрипторов: {len(self.current_frame.descriptors[0])}."
        )
        
        self._predict_pose()

        match, pts_2d, pts_3d = self._track_local_map_mono()

        success, R, t, inliers = self._optimize_pose_mono(pts_2d, pts_3d)       
        
        if not self._is_tracking_good(
            sucscess_pnp=success,
            num_inliers=inliers,
            num_matches=len(match)
        ):
            self.logger.warning("[Tracking] Потеря трекинга — переход в LOST.")
            self.state = State.LOST
            return self.state

        if self._need_new_keyframe(len(match)):
            self.logger.info("[Tracking-MONO] Создание нового keyframe.")
            self._create_keyframe(self.current_frame)
            
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
            camera=self.camera,
            images=[image],
            images_gray=[image_gray],
            keypoints=[kps],
            descriptors=[des],
        )        
        self.map.add_frame(frame)
        self.logger.info(f"[Tracking-MONO] Извлечено {len(kps)} фич.")
        return frame
    
    def _track_local_map_mono(self):
        
        f_cur = self.current_frame
        kf_ref = self.reference_keyframe

        # ------------------------------------------------------------
        # 1. Матчим дескрипторы
        # ------------------------------------------------------------
        des_cur = f_cur.descriptors[0]
        des_ref = kf_ref.descriptors[0]

        kps_cur = f_cur.keypoints[0]
        kps_ref = kf_ref.keypoints[0]

        matches = self.feature_tracker.match(des_ref, des_cur)
        self.logger.info(f"[Tracking-MONO] Найдено {len(matches)} исходных совпадений.")

        # ------------------------------------------------------------
        # 2. Фильтрация матчей 
        # ------------------------------------------------------------
        matches = filter_matches(
            matches=matches,
            kps1=kps_ref,
            kps2=kps_cur,
            logger=self.logger,
            **self.config.features.filtered_params
        )
        self.logger.info(f"[Tracking-MONO] После фильтрации осталось {len(matches)} совпадений.")

        if len(matches) < self.config.tracking.min_inliers:
            self.logger.warning(
                f"[Tracking-MONO] Недостаточно совпадений ({len(matches)} < {self.config.tracking.min_inliers})"
            )

        # ------------------------------------------------------------
        # 3. Формирование 3D-2D соответствий
        # ------------------------------------------------------------
        pts_2d: list[tuple[float, float]] = []
        pts_3d: list[np.ndarray] = []
        valid_matches: list[cv2.DMatch] = []

        for m in matches:
            idx_ref = m.queryIdx
            idx_cur = m.trainIdx

            # Получаем MapPoint из reference keyframe
            mp = kf_ref.get_map_point(cam_id=0, kp_idx=idx_ref)
            if mp is None or not mp.is_valid():
                continue

            # 3D координата точки
            pts_3d.append(mp.position)

            # 2D координата на текущем кадре
            pts_2d.append(kps_cur[idx_cur].pt)

            # Сохраняем совпадение
            valid_matches.append(m)

        num_valid = len(pts_3d)
        self.logger.info(f"[Tracking-MONO] Валидных 3D-точек: {num_valid}")

        if num_valid < 5:
            self.logger.warning("[Tracking-MONO] Очень мало 3D-точек для PnP.")

        # ------------------------------------------------------------
        # 4. Преобразуем в numpy
        # ------------------------------------------------------------
        pts_2d = np.array(pts_2d, dtype=np.float32) if num_valid > 0 else None
        pts_3d = np.array(pts_3d, dtype=np.float32) if num_valid > 0 else None

        return valid_matches, pts_2d, pts_3d
    
    def _optimize_pose_mono(
        self, 
        pts_2d: list[tuple[float, float]], 
        pts_3d: list[np.ndarray]
    ):
        if pts_3d is None or pts_2d is None or len(pts_3d) < 4:
            self.logger.warning("[Tracking-MONO] Недостаточно 3D-точек для PnP.")
            return False, None, None, 0
        
        self.logger.info(
            f"[Tracking-MONO] PnP: старт оптимизации. "
            f"  Точек: {len(pts_3d)}, "
        )

        K = self.camera.K
        dist = self.camera.dist
        use_ransac = self.config.tracking.use_ransac

        R, t, success, inliers_mask = estimate_motion_from_2d3d(
            pts3d=pts_3d,
            pts2d=pts_2d,
            K=K,
            dist=dist,
            use_ransac=use_ransac,
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
        self.logger.debug(f"[Tracking-MONO] Вектор t:\nS{t.reshape(-1)}")


        self.current_frame.set_pose_Rt(R, t)

        if self.config.debug:            
            t_w2c_ref = self.reference_keyframe.t_w2c
            t_c2w_ref = self.reference_keyframe.t_c2w
            
            self.logger.info(
                "[Initializer] Поза предыдущей камеры (global reference frame):"
                f" t_c2w = ({t_c2w_ref[0]:.4f}, {t_c2w_ref[1]:.4f}, {t_c2w_ref[2]:.4f})"
            )

            self.logger.info(
                "[Initializer] Поза предыдущей камеры (camera reference frame):"
                f" t_w2c = ({t_w2c_ref[0]:.4f}, {t_w2c_ref[1]:.4f}, {t_w2c_ref[2]:.4f})"
            )   
            
            t_w2c_cur = self.current_frame.t_w2c
            t_c2w_cur = self.current_frame.t_c2w
            
            self.logger.info(
                "[Initializer] Поза текущей камеры (global current frame):"
                f"  t_c2w = ({t_c2w_cur[0]:.4f}, {t_c2w_cur[1]:.4f}, {t_c2w_cur[2]:.4f})"
            )

            self.logger.info(
                "[Initializer] Поза текущей камеры (camera current frame):"
                f"  t_w2c = ({t_w2c_cur[0]:.4f}, {t_w2c_cur[1]:.4f}, {t_w2c_cur[2]:.4f})"
            )  

        if inliers_mask is not None:
            inlier_ratio = num_inliers / len(pts_2d)
            self.logger.info(
                f"[Tracking-MONO] PnP инлайеры: {num_inliers}/{len(pts_2d)} "
                f"({inlier_ratio*100:.2f}%)."
            )
        else:
            self.logger.warning("[Tracking-MONO] PnP: маска инлайеров отсутствует.")

        return True, R, t, num_inliers
        
    def _track_stereo(
        self,
        images: List[np.ndarray],
        timestamp: float,
    ):
        pass
    
    def _create_frame_stereo(
        self, 
        image_left: np.ndarray, 
        image_right: np.ndarray,
        timestamp: float
    ) -> Frame:
        pass
    
    def _track_rgbd(
        self,
        images: List[np.ndarray],
        timestamp: float,
        depth: np.ndarray,
    ):
        pass
    
    def _create_frame_rgbd(
        self, 
        image: np.ndarray, 
        timestamp: float, 
        depth: np.ndarray
    ) -> Frame:
        pass

    def _predict_pose(self):
        if self.last_frame is None:
            self.current_frame.pose = self.reference_keyframe.pose.copy()
            return
        self.current_frame.pose = self.last_frame.pose @ self.motion_model
        self.logger.info("[Tracking] Предсказание позы через motion model.")

    def _is_tracking_good(
        self,
        sucscess_pnp: bool,
        num_inliers: int,
        num_matches: int,
        reprojection_error: Optional[float] = None
    ) -> bool:
        
        self.logger.info("[Is Tracking good] Начало оценки качества трекинга.")

        cfg = self.config.tracking
        min_inliers = cfg.min_inliers
        min_inlier_ratio = cfg.min_inlier_ratio
        max_reprojection_error = cfg.max_reprojection_error
        
        # --------------------------------------------
        # 1. Проверка PnP
        # --------------------------------------------
        
        if not sucscess_pnp:
            self.logger.warning("[Is Tracking good] PnP неуспешен.")
            return False
        
        self.logger.info("[Is Tracking good] Проверка на PnP пройдена.")
        
        # --------------------------------------------
        # 2. Проверка числа инлайеров
        # --------------------------------------------
        
        if num_inliers < min_inliers:
            self.logger.warning(
                f"[Is Tracking good] Мало инлайеров ({num_inliers} < {min_inliers})"
            )
            return False
        
        self.logger.info(f"[Is Tracking good] Проверка количества инлайеров пройдена: {num_inliers} > {min_inliers}.")
        
        # --------------------------------------------
        # 3. Проверка отношения инлайеров к матчам
        # --------------------------------------------
        if num_matches > 0:
            ratio = num_inliers / num_matches
            if ratio < min_inlier_ratio:
                self.logger.warning(
                    f"[TrackingQuality] Низкое отношение инлайеров ({ratio:.2f} < {min_inlier_ratio})"
                )
                return False
        
        self.logger.info(f"[Is Tracking good] Проверка отношения инлайеров к матчам пройдена: {ratio:.2f} > {min_inlier_ratio}.")
        
        # --------------------------------------------
        # 4. Проверка ошибки репроекции
        # --------------------------------------------
        if reprojection_error is not None:
            if reprojection_error > max_reprojection_error:
                self.logger.warning(
                    f"[TrackingQuality] Большая ошибка репроекции ({reprojection_error:.2f} > {max_reprojection_error})"
                )
                return False
            
            self.logger.info(f"[Is Tracking good] Проверка ошибки репроекции пройдена: {reprojection_error:.2f} < {max_reprojection_error}.")

        self.logger.info("[Is Tracking good] Трекинг устойчив.")
        return True
    
    def _need_new_keyframe(
        self,
        matches_size
    ) -> bool:
        cfg = self.config.tracking

        # ============================================================
        # 1. Интервал между keyframes
        # ============================================================
        if self.reference_keyframe is not None:
            delta_frames = self.current_frame.id - self.reference_keyframe.id
            if delta_frames > cfg.keyframe_interval:
                self.logger.info(
                    f"[Need KeyFrame] Интервал {delta_frames} > {cfg.keyframe_interval}. Добавляем keyframe."
                )
                return True
            
        self.logger.info(f"[Need KeyFrame] Интервал между keyframes в норме: {delta_frames} / {cfg.keyframe_interval}.")

        # ============================================================
        # 2. Смещение камеры
        # ============================================================
        if self.last_frame is not None:

            T_last = self.last_frame.T_w2c
            T_cur  = self.current_frame.T_w2c

            dT = np.linalg.inv(T_last) @ T_cur

            # Трансляция
            trans = np.linalg.norm(dT[:3, 3])

            # Угол вращения
            rot_cos = (np.trace(dT[:3, :3]) - 1) / 2.0
            rot_cos = np.clip(rot_cos, -1.0, 1.0)
            rot = np.degrees(np.arccos(rot_cos))

            if trans > cfg.max_translation_for_kf:
                self.logger.info(
                    f"[Need KeyFrame] Большое смещение: Δt={trans:.3f} > {cfg.max_translation_for_kf}"
                )
                return True

            if rot > cfg.max_rotation_for_kf:
                self.logger.info(
                    f"[Need KeyFrame] Большой поворот: ΔR={rot:.2f}° > {cfg.max_rotation_for_kf}"
                )
                return True
        
        self.logger.info(f"[Need KeyFrame] Смещение камеры в норме: Δt={trans:.3f} < {cfg.max_translation_for_kf}, ΔR={rot:.2f}° < {cfg.max_rotation_for_kf}.")

        # ============================================================
        # 3. Количество матчей
        # ============================================================
        if matches_size < cfg.min_matches_for_kf:
            self.logger.info(
                f"[Need KeyFrame] Мало валидных матчей ({matches_size} < {cfg.min_matches_for_kf}) "
            )
            return True
        
        self.logger.info(f"[Need KeyFrame] Количество матчей в норме: {matches_size} / {cfg.min_matches_for_kf}.")
    
        return False

    def _create_keyframe(self, frame: Frame) -> KeyFrame:
        keyframe = KeyFrame.from_frame(frame)
        self.local_mapping.insert_keyframe(keyframe)
        return keyframe

    def _update_tracking_state(self):
        if self.last_frame is None or self.current_frame is None:
            self.motion_model = Pose()
            return
        T_last = self.last_frame.T_w2c
        T_cur  = self.current_frame.T_w2c        
        T_rel = np.linalg.inv(T_last) @ T_cur
        self.motion_model = Pose(T=T_rel)

    def _relocalize(self, images, timestamp):
        self.logger.warning("[Tracking] Попытка релокализации.")
        return None
