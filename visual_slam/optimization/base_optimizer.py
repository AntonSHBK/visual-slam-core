from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from visual_slam.config import Config
from visual_slam.utils.logging import get_logger
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint
from visual_slam.map.map import Map


class BaseOptimizer(ABC):
    def __init__(
        self,
        K: np.ndarray,
        config: Optional[Config] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.K: np.ndarray = K
        self.config: Config = config or Config()
        self.logger = logger or get_logger(self.__class__.__name__)

    @abstractmethod
    def optimize_initial(
        self,
        frame_ref: KeyFrame,
        frame_cur: KeyFrame,
        matches: List[Any],
    ) -> bool:
        pass

    @abstractmethod
    def optimize_local(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        pass

    @abstractmethod
    def optimize_global(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        pass

    @abstractmethod
    def optimize_pose_graph(
        self,
        slam_map: Map,
    ) -> bool:
        pass

 
    def compute_reprojection_error(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> float:
        pass

    def project_points(
        self,
        points_3d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        pass

    def log_stats(
        self,
        name: str,
        loss_before: float,
        loss_after: float,
        iterations: int,
    ) -> None:
        pass
