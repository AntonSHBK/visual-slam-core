import cv2
import numpy as np
from collections import defaultdict


def convert_matches_to_array_of_tuples(matches):
    """Конвертация cv2.DMatch в простые кортежи"""
    if isinstance(matches[0][0], cv2.DMatch):
        matches = [
            (
                (m[0].queryIdx, m[0].trainIdx, m[0].imgIdx, m[0].distance),
                (m[1].queryIdx, m[1].trainIdx, m[1].imgIdx, m[1].distance)
            )
            for m in matches if len(m) == 2
        ]
    return matches


def good_matches_one_to_one_py(matches, des1, des2, ratio_test=0.7):
    """Ratio test + уникальность trainIdx"""
    idxs1, idxs2 = [], []
    if matches is not None:
        float_inf = float('inf')
        dist_match = defaultdict(lambda: float_inf)
        index_match = dict()
        for m, n in matches:
            if m.distance > ratio_test * n.distance:
                continue
            dist = dist_match[m.trainIdx]
            if dist == float_inf:
                dist_match[m.trainIdx] = m.distance
                idxs1.append(m.queryIdx)
                idxs2.append(m.trainIdx)
                index_match[m.trainIdx] = len(idxs2) - 1
            else:
                if m.distance < dist:
                    index = index_match[m.trainIdx]
                    assert idxs2[index] == m.trainIdx
                    idxs1[index] = m.queryIdx
                    idxs2[index] = m.trainIdx
    return np.array(idxs1), np.array(idxs2)


def good_matches_simple_py(matches, des1, des2, ratio_test=0.7):
    """Простой ratio test (может давать дубли)"""
    idxs1, idxs2 = [], []
    if matches is not None:
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                idxs1.append(m.queryIdx)
                idxs2.append(m.trainIdx)
    return np.array(idxs1), np.array(idxs2)


def row_matches_py(matcher, kps1, des1, kps2, des2,
                   max_matching_distance, max_row_distance=2, max_disparity=100):
    """Фильтрация матчей по эпиполярному ограничению (для стерео)"""
    idxs1, idxs2 = [], []
    matches = matcher.match(np.array(des1), np.array(des2))
    for m in matches:
        pt1 = kps1[m.queryIdx]
        pt2 = kps2[m.trainIdx]
        if (m.distance < max_matching_distance and
            abs(pt1[1] - pt2[1]) < max_row_distance and
            abs(pt1[0] - pt2[0]) < max_disparity):
            idxs1.append(m.queryIdx)
            idxs2.append(m.trainIdx)
    return np.array(idxs1), np.array(idxs2)


def row_matches_with_ratio_test_py(matcher, kps1, des1, kps2, des2,
                                   max_matching_distance, max_row_distance=2,
                                   max_disparity=100, ratio_test=0.7):
    """Эпиполярное ограничение + ratio test"""
    idxs1, idxs2 = [], []
    matches = matcher.knnMatch(np.array(des1), np.array(des2), k=2)
    for m, n in matches:
        pt1 = kps1[m.queryIdx]
        pt2 = kps2[m.trainIdx]
        if (m.distance < max_matching_distance and
            abs(pt1[1] - pt2[1]) < max_row_distance and
            abs(pt1[0] - pt2[0]) < max_disparity):
            if m.distance < ratio_test * n.distance:
                idxs1.append(m.queryIdx)
                idxs2.append(m.trainIdx)
    return np.array(idxs1), np.array(idxs2)


def filter_non_row_matches_py(kps1, idxs1, kps2, idxs2,
                              max_row_distance=2, max_disparity=100):
    """Удаляет пары, не удовлетворяющие эпиполярным ограничениям"""
    assert len(idxs1) == len(idxs2)
    out_idxs1, out_idxs2 = [], []
    for idx1, idx2 in zip(idxs1, idxs2):
        pt1 = kps1[idx1]
        pt2 = kps2[idx2]
        if abs(pt1[1] - pt2[1]) < max_row_distance and abs(pt1[0] - pt2[0]) < max_disparity:
            out_idxs1.append(idx1)
            out_idxs2.append(idx2)
    return np.array(out_idxs1), np.array(out_idxs2)


def match_with_crosscheck_and_model_fit(matcher, des1, des2, kps1, kps2,
                                        ratio_test=0.7, cross_check=True,
                                        err_thld=1, info=''):
    """Фильтрация матчей с cross-check и RANSAC по фундаментальной матрице"""
    idxs1, idxs2 = [], []
    init_matches1 = matcher.knnMatch(des1, des2, k=2)
    init_matches2 = matcher.knnMatch(des2, des1, k=2)

    good_matches = []
    for i, (m1, n1) in enumerate(init_matches1):
        if cross_check and init_matches2[m1.trainIdx][0].trainIdx != i:
            continue
        if ratio_test is not None and m1.distance > ratio_test * n1.distance:
            continue
        good_matches.append(m1)
        idxs1.append(m1.queryIdx)
        idxs2.append(m1.trainIdx)

    # Преобразуем keypoints в numpy
    if isinstance(kps1, list) and isinstance(kps2, list):
        good_kps1 = np.array([kps1[m.queryIdx].pt for m in good_matches])
        good_kps2 = np.array([kps2[m.trainIdx].pt for m in good_matches])
    elif isinstance(kps1, np.ndarray) and isinstance(kps2, np.ndarray):
        good_kps1 = np.array([kps1[m.queryIdx] for m in good_matches])
        good_kps2 = np.array([kps2[m.trainIdx] for m in good_matches])
    else:
        raise TypeError("Keypoint type must be list[cv2.KeyPoint] or np.ndarray")

    # Оценка фундаментальной матрицы
    ransac_method = getattr(cv2, "USAC_MSAC", cv2.RANSAC)
    _, mask = cv2.findFundamentalMat(good_kps1, good_kps2,
                                     method=ransac_method,
                                     ransacReprojThreshold=err_thld,
                                     confidence=0.999)
    n_inlier = np.count_nonzero(mask)
    print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)

    return np.array(idxs1), np.array(idxs2), good_matches, mask
