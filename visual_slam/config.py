import os
import json
from typing import Any, Dict, Optional


class Config:
    """
    Класс конфигурации для алгоритма SLAM.
    Хранит все основные параметры (камера, датасет, SLAM-модули).
    """

    def __init__(self, config_path: Optional[str] = None):
        # --------------------
        # Общие настройки
        # --------------------
        self.verbose: bool = True
        self.save_logs: bool = True
        self.output_dir: str = "./output"

        # --------------------
        # Камера
        # --------------------
        self.camera_type: str = "pinhole"   # [pinhole, fisheye, rgbd]
        self.camera_width: int = 640
        self.camera_height: int = 480
        self.camera_fx: float = 525.0
        self.camera_fy: float = 525.0
        self.camera_cx: float = 320.0
        self.camera_cy: float = 240.0
        self.camera_dist_coeffs: list[float] = [0, 0, 0, 0, 0]
        self.camera_fps: int = 30
        self.camera_baseline: Optional[float] = None  # для стерео

        # --------------------
        # Датасет / источник данных
        # --------------------
        self.dataset_type: str = "images"   # [images, video, camera]
        self.dataset_path: str = "./data"
        self.sensor_type: str = "monocular" # [monocular, stereo, rgbd]

        # --------------------
        # Трекинг / фичи
        # --------------------
        self.feature_type: str = "ORB"      # [ORB, SIFT, SuperPoint, etc.]
        self.num_features: int = 1000
        self.use_motion_model: bool = True
        self.use_essential_matrix: bool = True

        # --------------------
        # Loop Closing
        # --------------------
        self.enable_loop_closing: bool = True
        self.loop_closing_method: str = "DBOW3"

        # --------------------
        # Bundle Adjustment
        # --------------------
        self.enable_local_BA: bool = True
        self.enable_global_BA: bool = True
        self.BA_window_size: int = 10

        # --------------------
        # Семантическая карта (опционально)
        # --------------------
        self.enable_semantics: bool = False
        self.semantic_model: str = "None"

        # --------------------
        # Дополнительно
        # --------------------
        self.far_points_threshold: Optional[float] = None
        self.use_fov_centers_based_kf_generation: bool = False
        self.max_fov_centers_distance: float = -1

        # Если задан путь к файлу → загрузить поверх дефолтов
        if config_path is not None:
            self.load(config_path)

    # --------------------
    # Методы
    # --------------------
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return self.__dict__

    def save(self, path: str):
        """Сохранение конфигурации в JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path: str):
        """Загрузка конфигурации из JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Config] Warning: unknown parameter {key}")

    def __repr__(self):
        return f"Config({json.dumps(self.to_dict(), indent=2)})"
