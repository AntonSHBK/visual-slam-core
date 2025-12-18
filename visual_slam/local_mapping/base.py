from abc import ABC, abstractmethod
from typing import Tuple, List
from logging import Logger
from typing import TYPE_CHECKING

from visual_slam.config import Config
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint

if TYPE_CHECKING:
    from visual_slam.slam import SLAM
    from visual_slam.camera import Camera    
    from visual_slam.map.map import Map
    

class BaseKeyframeHandler(ABC):

    def __init__(
        self, 
        slam: "SLAM", 
        config: "Config", 
        logger: "Logger"
    ):
        self.slam = slam
        self.config = config
        self.logger = logger
        
    @property
    def map(self) -> "Map":
        return self.slam.map
    
    @property
    def camera(self) -> "Camera":
        return self.slam.camera

    @abstractmethod
    def process_keyframe(self, kf: KeyFrame) -> Tuple[KeyFrame, List[MapPoint]]:
        pass
