from typing import List, Optional

import cv2
import numpy as np

from visual_slam.feature.base import BaseMatcher


# ==================================
# Brute Force Matcher (Hamming)
# ==================================
class BFMatcherHamming(BaseMatcher):
    def __init__(
        self,
        cross_check: bool = True,
        ratio_test: float = 0.75,
        **kwargs
    ):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check, **kwargs)
        self.ratio_test = ratio_test

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good: List[cv2.DMatch] = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good


# ==================================
# Brute Force Matcher (L2)
# ==================================
class BFMatcherL2(BaseMatcher):
    def __init__(
        self,
        cross_check: bool = True,
        ratio_test: float = 0.75,
        **kwargs
    ):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check, **kwargs)
        self.ratio_test = ratio_test

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good: List[cv2.DMatch] = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good


# ==================================
# FLANN Matcher
# ==================================
class FlannMatcher(BaseMatcher):
    def __init__(
        self,
        ratio_test: float = 0.75,
        **kwargs
    ):
        # Индекс для float-дескрипторов (KD-Tree)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params, **kwargs)
        self.ratio_test = ratio_test

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        # FLANN требует float32
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good: List[cv2.DMatch] = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good
