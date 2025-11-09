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
        cross_check: bool = False,
        ratio_thresh: float = 0.75,
        **kwargs
    ):  
        self.cross_check = cross_check
        self.ratio_thresh = ratio_thresh
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check, **kwargs)

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if self.cross_check:
            matches = self.matcher.match(desc1, desc2)
            return matches
        else:
            knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = [m for m, n in knn_matches if m.distance < self.ratio_thresh * n.distance]
            return good


# ==================================
# Brute Force Matcher (L2)
# ==================================
class BFMatcherL2(BaseMatcher):
    def __init__(
        self,
        cross_check: bool = False,
        ratio_thresh: float = 0.75,
        **kwargs
    ):
        self.cross_check = cross_check
        self.ratio_thresh = ratio_thresh
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check, **kwargs)

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if self.cross_check:
            matches = self.matcher.match(desc1, desc2)
            return matches
        else:
            knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = [m for m, n in knn_matches if m.distance < self.ratio_thresh * n.distance]
            return good
        

# ==================================
# FLANN Matcher
# ==================================
class FlannMatcher(BaseMatcher):
    def __init__(
        self,
        ratio_thresh: float = 0.75,
        **kwargs
    ):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params, **kwargs)
        self.ratio_thresh = ratio_thresh

    def match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray],
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)

        knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in knn_matches if m.distance < self.ratio_thresh * n.distance]
        return good
