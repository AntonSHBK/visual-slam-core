from typing import List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np

# ============================================================
# 1. Ratio Test (Lowe, 2004)
# ============================================================
def filter_matches_ratio(
    matcher: cv2.DescriptorMatcher,
    des1: np.ndarray,
    des2: np.ndarray,
    ratio_thresh: float = 0.75
) -> List[cv2.DMatch]:
    """
    Фильтрует матчи по правилу Lowe Ratio Test.
    
    Parameters
    ----------
    matcher : cv2.DescriptorMatcher
        Матчер (BFMatcher, FlannBased и т.п.)
    des1 : np.ndarray
        Дескрипторы эталонного кадра.
    des2 : np.ndarray
        Дескрипторы текущего кадра.
    ratio_thresh : float, optional
        Порог ratio-test (по умолчанию 0.75).
    
    Returns
    -------
    List[cv2.DMatch]
        Отфильтрованные матчи.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []

    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [
        m for m, n in knn_matches if m.distance < ratio_thresh * n.distance
    ]
    return good_matches

# ============================================================
# 2. Cross-check (взаимная проверка)
# ============================================================
def filter_matches_crosscheck(
    matcher: cv2.DescriptorMatcher,
    des1: np.ndarray,
    des2: np.ndarray,
    matches_12: List[List[cv2.DMatch]],
    k: int = 2
) -> List[cv2.DMatch]:
    """
    Cross-check фильтрация — оставляет только взаимные совпадения
    (когда A[i] ↔ B[j] и B[j] ↔ A[i]).

    Parameters
    ----------
    matcher : cv2.DescriptorMatcher
        Объект матчера.
    des1 : np.ndarray
        Дескрипторы эталонного кадра.
    des2 : np.ndarray
        Дескрипторы текущего кадра.
    matches_12 : List[List[cv2.DMatch]]
        Результат прямого матчинга A→B (обычно из knnMatch()).
    k : int, optional
        Количество ближайших соседей (по умолчанию 2).

    Returns
    -------
    List[cv2.DMatch]
        Отфильтрованные матчи, прошедшие взаимную проверку.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []

    matches_21 = matcher.knnMatch(des2, des1, k=k)

    cross_checked = []
    for i, pair in enumerate(matches_12):
        if len(pair) < 2:
            continue
        m1, _ = pair
        if matches_21[m1.trainIdx][0].trainIdx == i:
            cross_checked.append(m1)
    return cross_checked

# ============================================================
# 3. Геометрическая фильтрация (RANSAC по F-матрице)
# ============================================================
def filter_matches_ransac_fundamental(
    matches: List[cv2.DMatch],
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    ransac_thresh: float = 1.0,
    confidence: float = 0.999
) -> Tuple[List[cv2.DMatch], Optional[np.ndarray]]:
    """
    Геометрическая фильтрация матчей с помощью RANSAC по фундаментальной матрице.
    Удаляет выбросы, не согласующиеся с эпиполярной геометрией.

    Parameters
    ----------
    matches : List[cv2.DMatch]
        Исходные матчи.
    kps1, kps2 : List[cv2.KeyPoint]
        Ключевые точки эталонного и текущего кадров.
    ransac_thresh : float, optional
        Порог ошибки репроекции (в пикселях).
    confidence : float, optional
        Уровень доверия алгоритма RANSAC.

    Returns
    -------
    (inlier_matches, mask)
        Список инлайеров и бинарная маска (1 = инлайнер, 0 = аутлайер).
    """
    if len(matches) < 8:
        return [], None

    pts1 = np.array([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kps2[m.trainIdx].pt for m in matches])

    ransac_method = getattr(cv2, "USAC_MSAC", cv2.RANSAC)
    _, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=ransac_method,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence
    )

    if mask is None:
        return [], None

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, ok in zip(matches, mask) if ok]
    return inlier_matches, mask

# ============================================================
# 4. Orientation Consistency (ORB histogram)
# ============================================================
def filter_matches_with_histogram_orientation(
    matches: List[cv2.DMatch],
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    bins: int = 30
) -> List[cv2.DMatch]:
    """
    Отбрасывает матчи с несогласованной ориентацией (аналог ORB-SLAM).
    
    Parameters
    ----------
    matches : List[cv2.DMatch]
        Исходные матчи.
    kps1 : List[cv2.KeyPoint]
        Ключевые точки эталонного кадра.
    kps2 : List[cv2.KeyPoint]
        Ключевые точки текущего кадра.
    bins : int, optional
        Количество корзин в гистограмме углов.
    
    Returns
    -------
    List[cv2.DMatch]
        Отфильтрованные матчи.
    """
    if not matches:
        return matches

    # Разница углов ориентаций
    angles = np.array([
        kps1[m.queryIdx].angle - kps2[m.trainIdx].angle
        for m in matches
    ])
    # Приведение к диапазону [-180, 180]
    angles = (angles + 180) % 360 - 180

    hist, edges = np.histogram(angles, bins=bins, range=(-180, 180))
    max_bin = np.argmax(hist)
    bin_min, bin_max = edges[max_bin], edges[max_bin + 1]
    valid = (angles >= bin_min) & (angles < bin_max)

    return [m for m, ok in zip(matches, valid) if ok]


# ============================================================
# 5. Epipolar / Stereo Constraint
# ============================================================
def filter_matches_epipolar(
    matches: List[cv2.DMatch],
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    max_row_distance: float = 2.0,
    max_disparity: float = 100.0
) -> List[cv2.DMatch]:
    """
    Удаляет пары, не удовлетворяющие эпиполярному ограничению (для stereo).
    
    Parameters
    ----------
    matches : List[cv2.DMatch]
        Исходные матчи.
    kps1, kps2 : List[cv2.KeyPoint]
        Ключевые точки.
    max_row_distance : float
        Максимальное вертикальное отклонение между точками.
    max_disparity : float
        Максимальный сдвиг по X.
    
    Returns
    -------
    List[cv2.DMatch]
        Отфильтрованные матчи.
    """
    if not matches:
        return matches

    good_matches = []
    for m in matches:
        pt1 = kps1[m.queryIdx].pt
        pt2 = kps2[m.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_row_distance and abs(pt1[0] - pt2[0]) < max_disparity:
            good_matches.append(m)

    return good_matches

# ============================================================
# 6. Exclusion Mask Filter
# ============================================================
def filter_by_exclusion_mask(
    matches: List[cv2.DMatch],
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    mask_exclude_regions: List[Tuple[int, int, int, int]]
) -> List[cv2.DMatch]:
    """
    Исключает матчи, попадающие в запрещённые области изображения.
    Используется, например, чтобы убрать фичи с интерфейсных зон, HUD или текста.

    Parameters
    ----------
    matches : List[cv2.DMatch]
        Исходные матчи между кадрами.
    kps1, kps2 : List[cv2.KeyPoint]
        Ключевые точки эталонного и текущего кадров.
    mask_exclude_regions : list of tuple
        Список регионов [(xmin, ymin, xmax, ymax)], в которых матчи исключаются.

    Returns
    -------
    List[cv2.DMatch]
        Отфильтрованные матчи (вне запрещённых зон).
    """
    if not matches or not mask_exclude_regions:
        return matches

    filtered_matches: List[cv2.DMatch] = []

    for m in matches:
        x1, y1 = kps1[m.queryIdx].pt
        x2, y2 = kps2[m.trainIdx].pt

        excluded = any(
            (xmin <= x1 <= xmax and ymin <= y1 <= ymax) or
            (xmin <= x2 <= xmax and ymin <= y2 <= ymax)
            for (xmin, ymin, xmax, ymax) in mask_exclude_regions
        )

        if not excluded:
            filtered_matches.append(m)

    return filtered_matches

# ============================================================
# 7. Descriptor Distance Filter
# ============================================================
def filter_by_max_distance(
    matches: List[cv2.DMatch],
    max_distance: float = 50.0
) -> List[cv2.DMatch]:
    """
    Отбрасывает матчи с расстоянием между дескрипторами выше max_distance.

    Parameters
    ----------
    matches : List[cv2.DMatch]
        Исходные матчи между дескрипторами.
    max_distance : float, optional
        Максимально допустимое расстояние между совпадающими дескрипторами.

    Returns
    -------
    List[cv2.DMatch]
        Отфильтрованные матчи (только с distance <= max_distance).
    """
    if not matches:
        return matches

    filtered_matches = [m for m in matches if m.distance <= max_distance]
    return filtered_matches

# ============================================================
# 8. Unique Matches Filter
# ============================================================
def filter_unique_matches(
    matches: List[cv2.DMatch]
) -> List[cv2.DMatch]:
    """
    Удаляет дубликаты по trainIdx (оставляя только одно соответствие на каждую точку).
    Аналог removeDuplicatedMatches() из ORB-SLAM / PySLAM.

    Parameters
    ----------
    matches : List[cv2.DMatch]
        Список матчей (может содержать дубликаты trainIdx).

    Returns
    -------
    List[cv2.DMatch]
        Уникальные матчи (по trainIdx).
    """
    if not matches:
        return matches

    seen_train = set()
    unique = []
    for m in matches:
        if m.trainIdx not in seen_train:
            unique.append(m)
            seen_train.add(m.trainIdx)
    return unique


# ============================================================
# Общий интерфейс: filter_matches()
# ============================================================
def filter_matches(
    matches: List[cv2.DMatch],
    kps1: Optional[List[cv2.KeyPoint]] = None,
    kps2: Optional[List[cv2.KeyPoint]] = None,
    logger=None,
    **kwargs
) -> List[cv2.DMatch]:
    """
    Универсальный интерфейс фильтрации матчей.
    В зависимости от kwargs вызывает соответствующие фильтры.

    Параметры
    ----------
    matches : List[cv2.DMatch]
        Исходные матчи (может быть результат matcher.match или knnMatch).
    kps1, kps2 : list of cv2.KeyPoint, optional
        Ключевые точки для фильтров, которым нужны координаты (orientation, epipolar, model-fit).
    logger : logging.Logger, optional
        Логгер для вывода промежуточной информации.
    kwargs : dict
        Управляющие параметры и флаги фильтров.

    """

    if not matches:
        return []

    # --- Параметры и значения по умолчанию ---
    use_ransac_fund_matrix: bool = kwargs.get("use_ransac_fund_matrix", True)
    use_histogram_orientation: bool = kwargs.get("use_orientation", True)
    use_epipolar: bool = kwargs.get("use_epipolar", False)  
    use_max_distance: bool = kwargs.get("use_max_distance", False)
    use_exclusion_mask: bool = kwargs.get("use_exclusion_mask", False)
    use_unique: bool = kwargs.get("use_unique", False)  

    ransac_thresh: float = kwargs.get("ransac_thresh", 1.0)
    confidence: float = kwargs.get("confidence", 0.999)
    max_row_distance: float = kwargs.get("max_row_distance", 2.0)
    max_disparity: float = kwargs.get("max_disparity", 100.0)
    max_distance: float = kwargs.get("max_distance", 50.0)
    mask_exclude_regions = kwargs.get("mask_exclude_regions", [])

    # начальное количество
    filtered = matches
    total_before = len(filtered)
       
    # =====================================================
    # RANSAC Fundamental Matrix
    # =====================================================
    if use_ransac_fund_matrix and kps1 is not None and kps2 is not None:
        total_before = len(filtered)
        filtered, _ = filter_matches_ransac_fundamental(
            filtered,
            kps1,
            kps2,
            ransac_thresh=ransac_thresh,
            confidence=confidence,
        )
        if logger:
            logger.info(f"[FilterMatches] RANSAC (F-matrix): {total_before} → {len(filtered)}")

    # =====================================================
    # Orientation Histogram Filter
    # =====================================================
    if use_histogram_orientation and kps1 is not None and kps2 is not None:
        total_before = len(filtered)
        filtered = filter_matches_with_histogram_orientation(filtered, kps1, kps2)
        if logger:
            logger.info(f"[FilterMatches] Orientation histogram: {total_before} → {len(filtered)}")
        total_before = len(filtered)

    # =====================================================
    # Epipolar constraint (stereo only)
    # =====================================================
    if use_epipolar and kps1 is not None and kps2 is not None:
        total_before = len(filtered)
        filtered = filter_matches_epipolar(
            filtered,
            kps1,
            kps2,
            max_row_distance=max_row_distance,
            max_disparity=max_disparity,
        )
        if logger:
            logger.info(f"[FilterMatches] Epipolar filter: {total_before} → {len(filtered)}")
        total_before = len(filtered)

    if logger:
        logger.info(f"[FilterMatches] Финальное количество матчей: {len(filtered)}")
        
    # =====================================================
    # Descriptor distance filter
    # =====================================================
    if use_max_distance:
        total_before = len(filtered)
        filtered = filter_by_max_distance(filtered, max_distance=max_distance)
        if logger:
            logger.info(f"[FilterMatches] Max distance: {total_before} → {len(filtered)} (≤ {max_distance})")
        total_before = len(filtered)
    
    # =====================================================
    # Exclusion mask filter
    # =====================================================
    if use_exclusion_mask and mask_exclude_regions and kps1 is not None and kps2 is not None:
        total_before = len(filtered)
        filtered = filter_by_exclusion_mask(filtered, kps1, kps2, mask_exclude_regions)
        if logger:
            logger.info(f"[FilterMatches] Exclusion mask: {total_before} → {len(filtered)}")
        total_before = len(filtered)
        
    # =====================================================
    # Unique matches (remove duplicates)
    # =====================================================
    if use_unique:
        total_before = len(filtered)
        filtered = filter_unique_matches(filtered)
        if logger:
            logger.info(f"[FilterMatches] Unique matches: {total_before} → {len(filtered)}")
        total_before = len(filtered)

    return filtered
