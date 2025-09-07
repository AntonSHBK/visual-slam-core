
from visual_slam.slam import Slam


class Tracking:
  
    def __init__(self, slam: 'Slam'):
        self.slam = slam

    def track(self, frame):
        """Обновление состояния отслеживания."""
        pass
    
    def bundle_adjust(self):
        pass