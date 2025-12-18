from typing import List
from threading import Thread, Event
from abc import ABC, abstractmethod

from visual_slam.utils.logging import get_logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.map.map import Map
    from visual_slam.slam import SLAM
    from visual_slam.config import Config
    from visual_slam.camera import Camera
    from visual_slam.tracking import Tracking


class BaseHandler(ABC, Thread):

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

        self._stop_flag = False
        self.wakeup_event = Event()
        
    def trigger(self):
        """Вызывает событие."""
        self.wakeup_event.set()
        
    @property
    def map(self) -> "Map":
        return self.slam.map
    
    @property
    def camera(self) -> "Camera":
        return self.slam.camera
    
    @property
    def tracking(self) -> "Tracking":
        return self.slam.tracking

    def stop(self):
        self._stop_flag = True

    @abstractmethod
    def run(self):
        pass