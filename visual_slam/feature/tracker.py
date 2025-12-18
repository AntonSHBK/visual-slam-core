from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

from visual_slam.feature.feature_manager import FeatureManager
from visual_slam.utils.keypoints import filter_keypoints
from visual_slam.utils.matching import filter_matches
from visual_slam.viz.feature_viz import FeatureVisualizer


@dataclass
class FeatureTrackingResult:
    matches: List[cv2.DMatch]
    idxs1: List[int]
    idxs2: List[int]
    kps1_matched: np.ndarray  # shape (N,2), float32
    kps2_matched: np.ndarray  # shape (N,2), float32
    des1: Optional[np.ndarray]
    des2: Optional[np.ndarray]
    
    def filter_by_mask(self, mask: np.ndarray) -> "FeatureTrackingResult":
        mask = np.asarray(mask).astype(bool)
        if mask.ndim != 1 or mask.shape[0] != len(self.matches):
            raise ValueError(
                f"Mask shape {mask.shape} does not match number of matches {len(self.matches)}"
            )

        filtered_matches = [m for m, keep in zip(self.matches, mask) if keep]
        filtered_idxs1 = [i for i, keep in zip(self.idxs1, mask) if keep]
        filtered_idxs2 = [i for i, keep in zip(self.idxs2, mask) if keep]

        filtered_kps1 = self.kps1_matched[mask]
        filtered_kps2 = self.kps2_matched[mask]

        return FeatureTrackingResult(
            matches=filtered_matches,
            idxs1=filtered_idxs1,
            idxs2=filtered_idxs2,
            kps1_matched=filtered_kps1,
            kps2_matched=filtered_kps2,
            des1=self.des1,
            des2=self.des2,
        )
        

class BaseTracker(ABC):
    @abstractmethod
    def detectAndCompute(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        pass
    
    @abstractmethod
    def match(
        self, 
        img1: Optional[np.ndarray], 
        img2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        pass

    @abstractmethod
    def track(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kps1: List[cv2.KeyPoint],
        des2: Optional[np.ndarray],
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
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        return self.feature_manager.detectAndCompute(image, mask)
    
    def match(
            self, 
            des1: Optional[np.ndarray], 
            des2: Optional[np.ndarray]
        ) -> List[cv2.DMatch]:
            return self.feature_manager.match(des1, des2)
        
    def track(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kps1: Optional[List[cv2.KeyPoint]] = None,
        des1: Optional[np.ndarray] = None,
        kps2: Optional[List[cv2.KeyPoint]] = None,
        des2: Optional[np.ndarray] = None,
        logger=None,
        filtered_params: Optional[dict] = None
    ) -> FeatureTrackingResult:
        
        # viz = FeatureVisualizer(window_name_prefix="FeatureTracker")

        if kps2 is None or des2 is None:
            kps2, des2 = self.detectAndCompute(img2)
        
        if kps1 is None or des1 is None:
            kps1, des1 = self.detectAndCompute(img1)    
        
        if logger:
            logger.info(f"[Track] Обнаружено {len(kps2)} точек в текущем кадре.")
            logger.info(f"[Track] Обнаружено {len(kps1)} точек в эталонном кадре.")
        
        # viz.draw_keypoints(image_cur, kps_cur, window_name="Before filtering")
        
        kps2, des2 = filter_keypoints(
            kps=kps2,
            des=des2,
            logger=logger,
            **filtered_params
        )
        
        # viz.draw_keypoints(image_cur, kps_cur, window_name="After filtering")
        
        matches = self.match(des1, des2)
        if logger:
            logger.info(f"[Track] Найдено {len(matches)} исходных совпадений.")
        
        # viz.draw_matches(
        #     image_ref,
        #     image_cur,
        #     kps_ref,
        #     kps_cur,
        #     matches,
        #     window_name="Matches raw",
        #     max_display=None
        # )
        
        matches = filter_matches(
            matches,
            kps1=kps1,
            kps2=kps2,
            logger=logger,
            **filtered_params
        )   
        
        # viz.draw_matches(
        #     image_ref,
        #     image_cur,
        #     kps_ref,
        #     kps_cur,
        #     matches,
        #     window_name="Matches filtered",
        #     max_display=None
        # )    

        idxs1 = [m.queryIdx for m in matches]
        idxs2 = [m.trainIdx for m in matches]

        kps1_matched = np.array([kps1[i].pt for i in idxs1], dtype=np.float32)
        kps2_matched = np.array([kps2[i].pt for i in idxs2], dtype=np.float32)

        if logger:
            logger.info("[Track] Завершение трекинга")

        return FeatureTrackingResult(
            matches=matches,
            idxs1=idxs1,
            idxs2=idxs2,
            kps1_matched=kps1_matched,
            kps2_matched=kps2_matched,
            des1=des1,
            des2=des2,
        )
