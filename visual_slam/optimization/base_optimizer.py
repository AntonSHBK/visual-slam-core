from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from visual_slam.config import Config
from visual_slam.utils.logging import get_logger
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint


class BaseOptimizer(ABC):
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[Any] = None,
        log_dir: Optional[str] = 'logs',
    ) -> None:
        self.config: Config = config or Config()
        self.logger = logger or get_logger(
            self.__class__.__name__,
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )
 
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
    def optimize_initial(
        self,
        frame_ref: KeyFrame,
        frame_cur: KeyFrame,
        matches: List[Any],
    ) -> bool:
        pass
