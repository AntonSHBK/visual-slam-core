import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

from visual_slam.sensor_type import SensorType,  SensorItem

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    dist_coeffs: list = field(default_factory=lambda: [0, 0, 0, 0, 0])
    sensor_type: SensorItem = SensorType.MONOCULAR
    
    is_mono: bool = sensor_type == SensorType.MONOCULAR
    is_stereo: bool = sensor_type == SensorType.STEREO
    is_rgbd: bool = sensor_type == SensorType.RGBD
    

@dataclass
class FeatureConfig:
    detector: str = "orb"
    matcher: str = "bf-hamming"
    detector_params: Dict[str, Any] = field(default_factory=dict)
    matcher_params: Dict[str, Any] = field(default_factory=dict)
    filtered_params: Dict[str, Any] = field(default_factory=dict) 
    
@dataclass
class InitializationConfig:
    min_depth: float = 0.1
    max_depth: float = 100.0

@dataclass
class TrackingConfig:
    min_inliers: int = 10
    min_parallax_deg: float = 1.0
    keyframe_interval: int = 5
    min_inlier_ratio: float = 0.25
    max_reprojection_error: float = 5.0
    warmup_frames: int = 30
    extra: Dict[str, Any] = field(default_factory=dict)
    use_ransac: bool = True


@dataclass
class MappingConfig:
    local_ba_window: int = 5
    keyframe_insertion_thresh: float = 0.6
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    lr: float = 1e-3
    n_iter: int = 150

@dataclass
class LoopClosingConfig:
    pass

@dataclass
class AdditionalParamsConfig:
    ransac_threshold: float = 1.0
    prob: float = 0.999

@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loop_closing: LoopClosingConfig = field(default_factory=LoopClosingConfig)
    additional_params: AdditionalParamsConfig = field(default_factory=AdditionalParamsConfig)
    
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return asdict(self)

    def save(self, path: str):
        """Сохранение конфигурации в JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Загрузка конфигурации из JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data: dict = json.load(f)
        #  TODO: проверить все ли поля добавили
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            features=FeatureConfig(**data.get("features", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            mapping=MappingConfig(**data.get("mapping", {})),
            loop_closing=LoopClosingConfig(**data.get("loop_closing", {})),
            initialization=InitializationConfig(**data.get("initialization", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
        )

    def __repr__(self):
        return f"Config({json.dumps(self.to_dict(), indent=2)})"
