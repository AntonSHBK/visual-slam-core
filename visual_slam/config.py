import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

from visual_slam.sensor_type import SensorType,  SensorItem

@dataclass
class CameraConfig:
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
    max_reprojection_error: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)
    use_ransac: bool = True
    
    max_translation_for_kf: float = 2.0
    max_rotation_for_kf: float = 10.0
    min_matches_for_kf: int = 20
    
@dataclass
class LocalMappingConfig:
    run_timeout: float = 0.1
    max_neighbors: int = 2
    min_depth: float = 0.1
    max_depth: float = 100.0
    min_parallax_deg: float = 1.0

@dataclass
class MapConfig:
    pass

@dataclass
class OptimizationConfig:
    lr: float = 1e-3
    n_iter: int = 150
    batch_size: int = 1000
    huber_delta: float = 5.0

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
    local_mapping: LocalMappingConfig = field(default_factory=LocalMappingConfig)
    map: MapConfig = field(default_factory=MapConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loop_closing: LoopClosingConfig = field(default_factory=LoopClosingConfig)
    additional_params: AdditionalParamsConfig = field(default_factory=AdditionalParamsConfig)
    
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> "Config":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data: dict = json.load(f)
        #  TODO: проверить все ли поля добавил
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            features=FeatureConfig(**data.get("features", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            local_mapping=LocalMappingConfig(**data.get("local_mapping", {})),
            map=MapConfig(**data.get("map", {})),
            loop_closing=LoopClosingConfig(**data.get("loop_closing", {})),
            initialization=InitializationConfig(**data.get("initialization", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
        )

    def __repr__(self):
        return f"Config({json.dumps(self.to_dict(), indent=2)})"
