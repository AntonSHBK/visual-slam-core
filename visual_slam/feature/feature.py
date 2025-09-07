from typing import List, Optional, Tuple

import cv2
import numpy as np

from visual_slam.feature.base import BaseFeature2D


# =========================
# ORB
# =========================
class ORBFeature2D(BaseFeature2D):
    def __init__(
        self,
        nfeatures: int = 1000,
        scale_factor: float = 1.2,
        nlevels: int = 8,
        **kwargs
    ):
        self.orb = cv2.ORB.create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            **kwargs
        )

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[cv2.KeyPoint]:
        return self.orb.detect(image, mask)

    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.orb.compute(image, keypoints)

    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.orb.detectAndCompute(image, mask)


# =========================
# SIFT
# =========================
class SIFTFeature2D(BaseFeature2D):
    def __init__(
        self,
        nfeatures: int = 1000,
        **kwargs
    ):
        self.sift = cv2.SIFT.create(
            nfeatures=nfeatures,
            **kwargs
        )

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[cv2.KeyPoint]:
        return self.sift.detect(image, mask)

    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.sift.compute(image, keypoints)

    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.sift.detectAndCompute(image, mask)


# =========================
# FAST + BRIEF
# =========================
class FastBriefFeature2D(BaseFeature2D):
    def __init__(
        self,
        nfeatures: int = 500,
        **kwargs
    ):
        self.fast = cv2.FastFeatureDetector.create(**kwargs)

        try:
            self.brief = cv2.xfeatures2d.BriefDescriptorExtractor.create(bytes=32, **kwargs)
        except AttributeError:
            raise ImportError("OpenCV must be compiled with xfeatures2d (opencv-contrib-python) for BRIEF.")

        self.nfeatures = nfeatures

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[cv2.KeyPoint]:
        keypoints = self.fast.detect(image, mask)
        if len(keypoints) > self.nfeatures:
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:self.nfeatures]
        return keypoints

    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.brief.compute(image, keypoints)

    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        kps = self.detect(image, mask)
        return self.compute(image, kps)
