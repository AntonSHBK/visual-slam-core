from typing import List

from visual_slam.map.keyframe import KeyFrame
from .base_handler import BaseHandler

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.slam import SLAM
    from visual_slam.config import Config


class LocalHandler(BaseHandler):

    def __init__(
        self, 
        slam: "SLAM", 
        config: "Config", 
        log_dir="logs"
    ):
        super().__init__(
            slam=slam,
            config=config,
            log_dir=log_dir
        )
        self.run_timeout = self.config.local_mapping.run_timeout
        self.n_keyframe = self.config.local_mapping.max_neighbors
    
    def run(self):
        self.logger.info("[LocalHandler] Поток LocalHandler запущен.")

        while not self._stop_flag:
            self.wakeup_event.wait(timeout=self.run_timeout)

            if self._stop_flag:
                break

            keyframes = self.get_n_keyframes(self.n_keyframe)
            if len(keyframes) <= 2:
                self.wakeup_event.clear()
                continue

            self.local_optimize(keyframes)
            
            self.postprocessing()

            self.wakeup_event.clear()

        self.logger.info("[LocalHandler] Поток LocalHandler остановлен.")  
        
    def get_n_keyframes(self, n: int) -> List[KeyFrame]:
        keyframes = self.map.get_keyframes()[-n:]
        return keyframes

    def local_optimize(self, keyframes: List[KeyFrame]):
        
        mean_err_before_optimize = self.map.compute_mean_reprojection_error(
            camera=self.camera,
            keyframes=keyframes
        )
        self.logger.info(f"[Local Optimize] Средняя ошибка до оптимизации: {mean_err_before_optimize:.3f}px")
        
        self.logger.info("[Local Optimize] Начали оптимизацию")
        optimized = self.map.optimize_local(
            optimizer=self.tracking.optimizer,
            keyframes=keyframes
        )
        self.logger.info(f"[Local Optimize] Закончили оптимизацию: успешно={optimized}")

        mean_err_after_optimize = self.map.compute_mean_reprojection_error(
            camera=self.camera,
            keyframes=keyframes
        )
        self.logger.info(f"[Local Optimize] Средняя ошибка после оптимизации: {mean_err_after_optimize:.3f}px")
        pass
    
    def postprocessing(self):
        self.cull_bad_points()
    
    def cull_bad_points(self):
        pass
    
    def update_covisibility(self, kf: KeyFrame):
        pass
