import math
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

# =========================
# FAST_ORB_ANMS_Feature2D(
# =========================
class FastOrbAnmsFeature2D(BaseFeature2D):
    """
    FAST + ANMS Feature Detector:
    Быстрый детектор углов (FAST) с равномерным отбором точек через ANMS (SSC)
    и вычислением дескрипторов ORB (или BRIEF, если заменить).
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        threshold : int
            Порог FAST-детектора (по умолчанию 10).
        nonmaxSuppression : bool
            Включить внутреннее подавление немаксимумов (по умолчанию True).
        nfeatures : int
            Максимальное число фич для ORB-дескриптора.
        use_anms : bool
            Применять ли ANMS-отбор.
        anms_count : int
            Целевое количество точек после ANMS.
        anms_tolerance : float
            Допустимое отклонение при отборе ANMS.
        use_grid : bool
            Разделять ли изображение на сетку.
        grid_rows, grid_cols : int
            Размер сетки (по умолчанию 8×8).
        **kwargs : dict
            Дополнительные параметры пробрасываются в cv2.ORB.create().
        """
        # FAST-параметры
        self.threshold: int = kwargs.get("threshold", 10)
        self.nonmaxSuppression: bool = kwargs.get("nonmaxSuppression", True)

        # Grid-параметры
        self.use_grid: bool = kwargs.get("use_grid", False)
        self.grid_rows: int = kwargs.get("grid_rows", 8)
        self.grid_cols: int = kwargs.get("grid_cols", 8)

        # ANMS-параметры
        self.use_anms: bool = kwargs.get("use_anms", True)
        self.anms_count: int = kwargs.get("anms_count", 500)
        self.anms_tolerance: float = kwargs.get("anms_tolerance", 0.1)

        # ORB-параметры
        self.nfeatures: int = kwargs.get("nfeatures", 1000)

        # Инициализация детектора и дескриптора
        self.fast = cv2.FastFeatureDetector.create(
            threshold=self.threshold,
            nonmaxSuppression=self.nonmaxSuppression
        )
        self.orb = cv2.ORB.create(nfeatures=self.nfeatures)

    def detect(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> List[cv2.KeyPoint]:
        """
        Детектирует ключевые точки с использованием FAST и (опционально) сетки.
        """
        if not self.use_grid:
            keypoints = self.fast.detect(image, mask)
        else:
            keypoints = self._detect_grid(image)
        return keypoints

    def compute(
        self, image: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Вычисляет дескрипторы ORB для заданных keypoints.
        """
        if not keypoints:
            return [], None
        keypoints, descriptors = self.orb.compute(image, keypoints)
        return keypoints, descriptors

    def detectAndCompute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Комбинированный метод: FAST → (опционально Grid) → (опционально ANMS) → ORB.
        """
        keypoints = self.detect(image, mask)
        if not keypoints:
            return [], None

        if self.use_anms:
            h, w = image.shape[:2]
            keypoints = self._apply_anms(keypoints, w, h)

        return self.compute(image, keypoints)

    def _detect_grid(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        Делит изображение на сетку и применяет FAST в каждой ячейке.
        """
        h, w = image.shape[:2]
        kp_all: List[cv2.KeyPoint] = []
        rows, cols = self.grid_rows, self.grid_cols

        for i in range(rows):
            for j in range(cols):
                x0, x1 = int(j * w / cols), int((j + 1) * w / cols)
                y0, y1 = int(i * h / rows), int((i + 1) * h / rows)
                roi = image[y0:y1, x0:x1]
                kp = self.fast.detect(roi, None)
                for k in kp:
                    k.pt = (k.pt[0] + x0, k.pt[1] + y0)
                kp_all.extend(kp)
        return kp_all

    def _apply_anms(
        self, keypoints: List[cv2.KeyPoint], cols: int, rows: int
    ) -> List[cv2.KeyPoint]:
        """
        Применяет ANMS (Adaptive Non-Maximal Suppression) к списку keypoints.
        """
        num_ret_points = self.anms_count
        tolerance = self.anms_tolerance

        keypoints_sorted = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        idx_ssc = self._anms_ssc(keypoints_sorted, num_ret_points, tolerance, cols, rows)
        keypoints_anms = [keypoints_sorted[i] for i in idx_ssc]
        return keypoints_anms

    def _anms_ssc(
        self,
        keypoints: List[cv2.KeyPoint],
        num_ret_points: int,
        tolerance: float,
        cols: int,
        rows: int,
    ) -> List[int]:
        """
        Реализация SSC (Suppression via Square Covering).
        Возвращает индексы выбранных точек.
        """
        if len(keypoints) <= num_ret_points:
            return list(range(len(keypoints)))

        exp1 = rows + cols + 2 * num_ret_points
        exp2 = (
            4 * cols + 4 * num_ret_points + 4 * rows * num_ret_points
            + rows * rows + cols * cols - 2 * rows * cols
            + 4 * rows * cols * num_ret_points
        )
        exp3 = math.sqrt(exp2)
        exp4 = num_ret_points - 1

        sol1 = -round(float(exp1 + exp3) / exp4)
        sol2 = -round(float(exp1 - exp3) / exp4)
        high = sol1 if (sol1 > sol2) else sol2
        low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

        prev_width = -1
        result_list = []
        complete = False
        k_min = round(num_ret_points - (num_ret_points * tolerance))
        k_max = round(num_ret_points + (num_ret_points * tolerance))

        while not complete:
            width = low + (high - low) / 2
            if width == prev_width or low > high:
                break

            c = width / 2
            num_cell_cols = int(cols / c)
            num_cell_rows = int(rows / c)
            covered_vec = [[False] * (num_cell_cols + 1) for _ in range(num_cell_rows + 1)]
            result = []

            for i, kp in enumerate(keypoints):
                row = int(kp.pt[1] / c)
                col = int(kp.pt[0] / c)
                if not covered_vec[row][col]:
                    result.append(i)
                    row_min = max(0, int(row - width / c))
                    row_max = min(num_cell_rows, int(row + width / c))
                    col_min = max(0, int(col - width / c))
                    col_max = min(num_cell_cols, int(col + width / c))
                    for r in range(row_min, row_max + 1):
                        for c_ in range(col_min, col_max + 1):
                            covered_vec[r][c_] = True

            if k_min <= len(result) <= k_max:
                result_list = result
                complete = True
            elif len(result) < k_min:
                high = width - 1
            else:
                low = width + 1
            prev_width = width

        if not result_list:
            result_list = result
        return result_list
