from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

from visual_slam.feature.feature_manager import FeatureManager


@dataclass
class FeatureTrackingResult:
    matches: List[cv2.DMatch]
    idxs_ref: List[int]
    idxs_cur: List[int]
    kps_ref_matched: np.ndarray  # shape (N,2), float32
    kps_cur_matched: np.ndarray  # shape (N,2), float32
    des_ref: Optional[np.ndarray]
    des_cur: Optional[np.ndarray]


class BaseTracker(ABC):
    @abstractmethod
    def detectAndCompute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        pass

    @abstractmethod
    def track(
        self,
        image_ref: np.ndarray,
        image_cur: np.ndarray,
        kps_ref: List[cv2.KeyPoint],
        des_ref: Optional[np.ndarray],
    ) -> FeatureTrackingResult:
        pass


class FeatureTracker(BaseTracker):
    def __init__(
        self,
        detector: str = "orb",
        matcher: str = "bf-hamming",
        detector_params: Optional[dict] = None,
        matcher_params: Optional[dict] = None,
    ):
        self.feature_manager = FeatureManager(
            detector=detector,
            matcher=matcher,
            detector_params=detector_params,
            matcher_params=matcher_params,
        )

    def detectAndCompute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.feature_manager.detectAndCompute(image, mask)

    def track(
        self,
        image_ref: np.ndarray,
        image_cur: np.ndarray,
        kps_ref: List[cv2.KeyPoint],
        des_ref: Optional[np.ndarray],
    ) -> FeatureTrackingResult:
        kps_cur, des_cur = self.detectAndCompute(image_cur)
        matches = self.feature_manager.match(des_ref, des_cur)

        idxs_ref = [m.queryIdx for m in matches]
        idxs_cur = [m.trainIdx for m in matches]

        kps_ref_matched = np.array([kps_ref[i].pt for i in idxs_ref], dtype=np.float32)
        kps_cur_matched = np.array([kps_cur[i].pt for i in idxs_cur], dtype=np.float32)

        return FeatureTrackingResult(
            matches=matches,
            idxs_ref=idxs_ref,
            idxs_cur=idxs_cur,
            kps_ref_matched=kps_ref_matched,
            kps_cur_matched=kps_cur_matched,
            des_ref=des_ref,
            des_cur=des_cur,
        )
