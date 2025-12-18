from visual_slam.feature.base import BaseFeature2D, BaseMatcher
from visual_slam.feature.feature import (
    ORBFeature2D,
    SIFTFeature2D,
    FastBriefFeature2D,
    FastOrbAnmsFeature2D
)
from visual_slam.feature.matcher import (
    BFMatcherHamming,
    BFMatcherL2,
    FlannMatcher,
)


def feature_factory(name: str, **kwargs) -> BaseFeature2D:
    name = name.lower()
    if name == "orb":
        return ORBFeature2D(**kwargs)
    elif name in ("fast_orb_anms", "fast-orb-anms"):
        return FastOrbAnmsFeature2D(**kwargs)
    elif name == "sift":
        return SIFTFeature2D(**kwargs)
    elif name in ("fastbrief", "fast+brief", "fb"):
        return FastBriefFeature2D(**kwargs)
    else:
        raise ValueError(f"Неизвестный тип feature: {name}")    
    

def matcher_factory(name: str, **kwargs) -> BaseMatcher:
    name = name.lower()
    if name in ("bf_hamming", "bf-hamming", "hamming"):
        return BFMatcherHamming(**kwargs)
    elif name in ("bf-l2", "l2"):
        return BFMatcherL2(**kwargs)
    elif name == "flann":
        return FlannMatcher(**kwargs)
    else:
        raise ValueError(f"Неизвестный тип matcher: {name}")
    

class FeatureManager:
    """
    Управляет выбором детектора/дескриптора и матчера.
    """
    def __init__(
        self,
        detector: str = "orb",
        matcher: str = "bf-hamming",
        detector_params: dict = None,
        matcher_params: dict = None
    ):
        detector_params = detector_params or {}
        matcher_params = matcher_params or {}
        
        self.feature = feature_factory(detector, **detector_params)
        self.matcher = matcher_factory(matcher, **matcher_params)

    def detectAndCompute(self, image, mask=None):
        return self.feature.detectAndCompute(image, mask)

    def match(self, desc1, desc2):
        return self.matcher.match(desc1, desc2)
    
    def filter_keypoints(self, type, frame, kps, des=None):
        # TODO Разобраться с фильтрацией
        pass
