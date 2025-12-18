import threading
from typing import List
from threading import Thread
import queue
import time

from anyio import key
import numpy as np

from visual_slam.map.keyframe import KeyFrame
from visual_slam.utils.logging import get_logger
from visual_slam.local_mapping.mono import MonoKeyframeHandler
# from .stereo import StereoKeyframeHandler     # когда появятся
# from .rgbd import RGBDKeyframeHandler         # когда появятся


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.map.map import Map
    from visual_slam.map.map_point import MapPoint
    from visual_slam.slam import SLAM
    from visual_slam.config import Config
    from visual_slam.camera import Camera
    from visual_slam.tracking import Tracking


class LocalMapping(Thread):

    def __init__(
        self, 
        slam: "SLAM", 
        config: "Config", 
        log_dir="logs"
    ):
        super().__init__()
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )
        self.logger.info(f"\n{'*'*80}")
        self.slam = slam
        self.config = config

        self.new_keyframes = queue.Queue()
        self.timeout = self.config.local_mapping.run_timeout
        self._stop_flag = False
        
        if self.config.camera.is_mono:
            self.handler = MonoKeyframeHandler(slam, config, self.logger)
        elif self.config.camera.is_stereo:
            pass
            # self.handler = StereoKeyframeHandler(slam, config, self.logger)
        elif self.config.camera.is_rgbd:
            pass
            # self.handler = RGBDKeyframeHandler(slam, config, self.logger)
        else:
            raise ValueError("Unsupported camera type")
    
    @property
    def map(self) -> "Map":
        return self.slam.map
    
    @property
    def camera(self) -> "Camera":
        return self.slam.camera
    
    @property
    def tracking(self) -> "Tracking":
        return self.slam.tracking

    def insert_keyframe(self, kf: KeyFrame):
        self.new_keyframes.put(kf)
        self.logger.info(f"[LocalMapping] Добавлен KeyFrame id={kf.keyframe_id} в очередь.")

    def stop(self):
        self._stop_flag = True

    def run(self):
        self.logger.info("[LocalMapping] Поток запущен.")
        

        while not self._stop_flag:
            try:
                kf: KeyFrame = self.new_keyframes.get(timeout=self.timeout)
            except queue.Empty:
                # self.logger.info("[LocalMapping] Очередь KeyFrame пуста, ожидание...")
                continue

            self.logger.info(f"[LocalMapping] Обработка KeyFrame keyframe_id={kf.keyframe_id} из очереди.")
            self.process_keyframe(kf)

        self.logger.info("[LocalMapping] Поток остановлен.")

    def process_keyframe(self, kf: KeyFrame):
        
        self.logger.info(f"[LocalMapping] ======================== KF {kf.keyframe_id}")

        kf, new_map_points = self.handler.process_keyframe(kf)

        # 1. Добавить KF в карту
        self.add_keyframe_to_map(kf)
        
        # 2. Добавить MapPoints в глобальную карту
        self.add_points_to_map(new_map_points)        

        # 3. Обновить связи ковизибилити
        # self.update_covisibility(kf)

        # 4. Оптимизация: локальная         
        # keyframes = self.get_n_keyframes(
        #     self.config.local_mapping.max_neighbors + 1
        # )
        # self.local_optimize(keyframes)

        # 5. Очистка карты
        # self.cull_bad_points()

        self.logger.info(
            f"[LocalMapping] KF {kf.keyframe_id}: обработка завершена. "
            f"Добавлено точек: {len(new_map_points)}"
        )

    def add_keyframe_to_map(self, kf: KeyFrame):
        self.map.add_keyframe(kf)
        self.logger.info(f"[LocalMapping] Добавлен KeyFrame keyframe_id={kf.keyframe_id} в карту.")

    def add_points_to_map(self, map_points: List["MapPoint"]):
        """Добавить триангулированные точки в карту."""
        for mp in map_points:
            self.map.add_map_point(mp)
        self.logger.info(f"[LocalMapping] В карту добавлено точек: {len(map_points)}")
        
    # def get_n_keyframes(self, n: int) -> List[KeyFrame]:
    #     keyframes = self.map.get_keyframes()[-n:]
    #     return keyframes

    # def local_optimize(self, keyframes: List[KeyFrame]):
        
    #     mean_err_before_optimize = self.tracking.map.compute_mean_reprojection_error(
    #         camera=self.camera,
    #         keyframes=keyframes
    #     )
    #     self.logger.info(f"[Initializer] Средняя ошибка до оптимизации: {mean_err_before_optimize:.3f}px")
        
    #     self.logger.info("[Initializer] Начали оптимизацию")
    #     optimized = self.tracking.map.optimize_local(
    #         optimizer=self.tracking.optimizer,
    #         keyframes=keyframes
    #     )
    #     self.logger.info(f"[Initializer] Закончили оптимизацию: успешно={optimized}")

    #     mean_err_after_optimize = self.tracking.map.compute_mean_reprojection_error(
    #         camera=self.camera,
    #         keyframes=keyframes
    #     )
    #     self.logger.info(f"[Initializer] Средняя ошибка после оптимизации: {mean_err_after_optimize:.3f}px")
    #     pass
    
    # def cull_bad_points(self):
    #     pass
    
    # def update_covisibility(self, kf: KeyFrame):
    #     pass