from visual_slam.config import Config


class SlamBase:
    """
    Базовый класс для SLAM-систем.
    """

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        """Запуск процесса SLAM."""
        raise NotImplementedError("SLAM run method not implemented.")


class Slam(SlamBase):
    """
    Простой класс для SLAM-систем.
    """

    def __init__(self, config: Config):
        super().__init__(config)

    def run(self):
        """Запуск процесса SLAM."""
        print("Запуск простого SLAM...")
        
        
    def bundle_adjust(self):
        pass