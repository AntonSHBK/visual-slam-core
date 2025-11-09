from __future__ import annotations
from typing import Any, List, Optional, Dict

import numpy as np
import torch

from visual_slam.config import Config
from visual_slam.map.keyframe import KeyFrame
from visual_slam.map.map_point import MapPoint
from visual_slam.map.map import Map
from visual_slam.optimization.base_optimizer import BaseOptimizer


class TorchOptimizer(BaseOptimizer):
    def __init__(
        self,
        K: np.ndarray,
        config: Optional[Config] = None,
        logger: Optional[Any] = None,
    ) -> None:
        super().__init__(K, config, logger)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize_initial(
        self,
        frame_ref: KeyFrame,
        frame_cur: KeyFrame,
        matches: List[Any],
    ) -> bool:
        pass

    def optimize_local(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        pass

    def optimize_global(
        self,
        keyframes: List[KeyFrame],
        map_points: List[MapPoint],
    ) -> bool:
        pass

    def optimize_pose_graph(
        self,
        slam_map: Map,
    ) -> bool:
        pass

    def _build_tensors(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        pass

    def _compute_loss(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass

    def _run_optimization(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass

    def _update_parameters(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass
