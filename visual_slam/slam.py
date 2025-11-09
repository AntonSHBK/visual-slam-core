from visual_slam.map.map import Map
from visual_slam.state import State
from visual_slam.tracking import Tracking
from visual_slam.feature.tracker import FeatureTracker
from visual_slam.utils.logging import get_logger


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.camera import CameraBase
    from visual_slam.config import Config
    
class SLAM:
    def __init__(
        self, 
        camera: "CameraBase", 
        config: "Config",
        log_dir="logs"
    ):
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )
        
        self.state = State.NO_IMAGES_YET        
        
        self.camera = camera        
        self.config = config
        
        self.map = Map()   
             
        self.feature_tracker = FeatureTracker(
            detector=config.features.detector,
            matcher=config.features.matcher,
            detector_params=config.features.detector_params,
            matcher_params=config.features.matcher_params,
        )
        self.tracking = Tracking(self, config)

        # TODO: сюда же добавим LocalMapping и LoopClosing
        self.local_mapping = None
        self.loop_closing = None

        self.is_running = False

    def track(self, images, timestamp=None, depth=None):
        return self.tracking.track(images, timestamp, depth=depth)

    def reset(self):
        self.map.reset()
        self.tracking.reset()
        # TODO: сбросить и другие модули
