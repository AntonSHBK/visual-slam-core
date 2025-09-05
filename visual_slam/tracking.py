from visual_slam.config import Config


class Tracking:
  
    """
    Класс для отслеживания объектов.
    """

    def __init__(self, config: Config):
        self.config = config

    def track(self, frame):
        """Обновление состояния отслеживания."""
        pass