from .feature import ORBFeature2D, SIFTFeature2D, FastBriefFeature2D
from .matcher import BFMatcherHamming, BFMatcherL2, FlannMatcher


class FeatureFactory:
    """Фабрика для создания пайплайнов (детектор+дескриптор+матчер)."""

    @staticmethod
    def create(name="ORB_BF", nfeatures=1000, ratio_test=0.75):
        name = name.upper()

        # ORB + BF (Hamming)
        if name == "ORB_BF":
            feature2d = ORBFeature2D(nfeatures=nfeatures)
            matcher = BFMatcherHamming(ratio_test=ratio_test)
            return feature2d, matcher

        # SIFT + BF (L2)
        elif name == "SIFT_BF":
            feature2d = SIFTFeature2D(nfeatures=nfeatures)
            matcher = BFMatcherL2(ratio_test=ratio_test)
            return feature2d, matcher

        # SIFT + FLANN
        elif name == "SIFT_FLANN":
            feature2d = SIFTFeature2D(nfeatures=nfeatures)
            matcher = FlannMatcher(ratio_test=ratio_test)
            return feature2d, matcher

        # FAST + BRIEF + BF (Hamming)
        elif name == "FAST_BRIEF_BF":
            feature2d = FastBriefFeature2D(nfeatures=nfeatures)
            matcher = BFMatcherHamming(ratio_test=ratio_test)
            return feature2d, matcher

        else:
            raise ValueError(f"Unknown pipeline: {name}")
