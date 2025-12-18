from visual_slam.map.map import Map
from visual_slam.state import State
from visual_slam.tracking import Tracking
from visual_slam.feature.tracker import FeatureTracker
from visual_slam.utils.logging import get_logger
from visual_slam.local_mapping.local_mapping import LocalMapping
from visual_slam.handlers.local_handler import LocalHandler
# from visual_slam.handlers.global_handler import 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.camera import Camera
    from visual_slam.config import Config
    
class SLAM:
    def __init__(
        self, 
        camera: "Camera", 
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
             
        self.feature_tracker = FeatureTracker(
            detector=config.features.detector,
            matcher=config.features.matcher,
            detector_params=config.features.detector_params,
            matcher_params=config.features.matcher_params,
        )
        self.map = Map(max_frames=5)
        self.tracking = Tracking(self, config, log_dir=log_dir) 
        self.local_mapping = LocalMapping(self, config, log_dir=log_dir)
        self.local_handler = LocalHandler(self, config, log_dir=log_dir)
        self.loop_closing = None

        self.is_running = False
        
        self._post_start()
        
    def _post_start(self):
        self.local_mapping.start()
        self.local_handler.start()
        
        pass
        
    def shutdown(self):
        self.local_mapping.stop()
        self.local_mapping.join()
        
        self.local_handler.stop()
        self.local_handler.join()
        
        pass

    def track(self, images, timestamp=None, depth=None):
        return self.tracking.track(images, timestamp, depth=depth)

    def reset(self):
        self.map.reset()
        self.tracking.reset()
        # TODO: сбросить и другие модули
