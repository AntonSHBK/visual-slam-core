import cv2
from .base import BaseMatcher


# ==================================
# Brute Force Matcher (Hamming)
# Для бинарных дескрипторов (ORB, BRIEF, BRISK, AKAZE)
# ==================================
class BFMatcherHamming(BaseMatcher):
    def __init__(self, cross_check=True, ratio_test=0.75):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        self.ratio_test = ratio_test

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good


# ==================================
# Brute Force Matcher (L2)
# Для float-дескрипторов (SIFT, SURF, AKAZE float)
# ==================================
class BFMatcherL2(BaseMatcher):
    def __init__(self, cross_check=True, ratio_test=0.75):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
        self.ratio_test = ratio_test

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good


# ==================================
# FLANN Matcher
# Быстрее на больших float-дескрипторах (SIFT, SURF)
# ==================================
class FlannMatcher(BaseMatcher):
    def __init__(self, ratio_test=0.75):
        # Индекс для float-дескрипторов (KD-Tree)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.ratio_test = ratio_test

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        # FLANN требует float32
        if desc1.dtype != 'float32':
            desc1 = desc1.astype('float32')
        if desc2.dtype != 'float32':
            desc2 = desc2.astype('float32')

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good
