import cv2
from .base import BaseFeature2D


# =========================
# ORB
# =========================
class ORBFeature2D(BaseFeature2D):
    """ORB: объединяет детектор и дескриптор"""

    def __init__(self, nfeatures=1000, scale_factor=1.2, nlevels=8):
        self.orb = cv2.ORB.create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels
        )

    def detect(self, image, mask=None):
        return self.orb.detect(image, mask)

    def compute(self, image, keypoints):
        return self.orb.compute(image, keypoints)

    def detectAndCompute(self, image, mask=None):
        return self.orb.detectAndCompute(image, mask)


# =========================
# SIFT
# =========================
class SIFTFeature2D(BaseFeature2D):
    """SIFT: классический float-дескриптор, устойчивый к масштабу и повороту"""

    def __init__(self, nfeatures=1000):
        self.sift = cv2.SIFT.create(nfeatures=nfeatures)

    def detect(self, image, mask=None):
        return self.sift.detect(image, mask)

    def compute(self, image, keypoints):
        return self.sift.compute(image, keypoints)

    def detectAndCompute(self, image, mask=None):
        return self.sift.detectAndCompute(image, mask)


# =========================
# FAST + BRIEF
# =========================
class FastBriefFeature2D(BaseFeature2D):
    """Комбинация FAST (детектор) + BRIEF (дескриптор).
    FAST сам по себе не считает дескрипторы, поэтому берём BRIEF.
    """

    def __init__(self, nfeatures=500):
        # FAST — только детектор
        self.fast = cv2.FastFeatureDetector.create()

        # BRIEF — только дескриптор
        try:
            self.brief = cv2.xfeatures2d.BriefDescriptorExtractor.create(bytes=32)
        except AttributeError:
            raise ImportError("OpenCV must be compiled with xfeatures2d (opencv-contrib-python) for BRIEF.")

        self.nfeatures = nfeatures

    def detect(self, image, mask=None):
        keypoints = self.fast.detect(image, mask)
        if len(keypoints) > self.nfeatures:
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:self.nfeatures]
        return keypoints

    def compute(self, image, keypoints):
        return self.brief.compute(image, keypoints)

    def detectAndCompute(self, image, mask=None):
        kps = self.detect(image, mask)
        return self.compute(image, kps)
