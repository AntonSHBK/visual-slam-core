from typing import List, Tuple, Optional

import cv2
import numpy as np


import cv2
import numpy as np
from typing import List, Tuple, Optional

# -------------------------------------------------------
# 1. Фильтр: GRID распределение по изображению
# -------------------------------------------------------
def filter_keypoints_grid(
    kps: List[cv2.KeyPoint],
    des: Optional[np.ndarray],
    grid_size: Tuple[int, int] = (3, 3),
    max_per_cell: int = 100,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Ограничивает количество точек в каждой ячейке сетки.
    """
    if not kps:
        return kps, des

    # Определяем границы сетки
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    h, w = int(np.max(pts[:, 1]) + 1), int(np.max(pts[:, 0]) + 1)
    grid_w, grid_h = w // grid_size[0], h // grid_size[1]
    grid = [[[] for _ in range(grid_size[0])] for _ in range(grid_size[1])]

    # Раскладываем фичи по ячейкам
    for kp, d in zip(kps, des):
        gx, gy = int(kp.pt[0] // grid_w), int(kp.pt[1] // grid_h)
        gx = min(gx, grid_size[0] - 1)
        gy = min(gy, grid_size[1] - 1)
        grid[gy][gx].append((kp, d))

    filtered_kps, filtered_des = [], []
    for row in grid:
        for cell in row:
            if not cell:
                continue
            # выбираем top-N по response
            cell = sorted(cell, key=lambda x: -x[0].response)
            for kp, d in cell[:max_per_cell]:
                filtered_kps.append(kp)
                filtered_des.append(d)

    des_out = np.array(filtered_des) if filtered_des else None
    return filtered_kps, des_out


# -------------------------------------------------------
# 2. Фильтр: NMS (Non-Max Suppression)
# -------------------------------------------------------
def filter_keypoints_nms(
    kps: List[cv2.KeyPoint],
    des: Optional[np.ndarray],
    radius: int = 8,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Применяет non-max suppression, оставляя только наиболее сильные фичи в радиусе.
    """
    if not kps:
        return kps, des

    mask = np.ones(len(kps), dtype=bool)
    for i in range(len(kps)):
        if not mask[i]:
            continue
        for j in range(i + 1, len(kps)):
            if not mask[j]:
                continue
            if np.linalg.norm(np.array(kps[i].pt) - np.array(kps[j].pt)) < radius:
                if kps[i].response >= kps[j].response:
                    mask[j] = False
                else:
                    mask[i] = False

    kps_out = [kp for i, kp in enumerate(kps) if mask[i]]
    des_out = des[mask] if des is not None else None
    return kps_out, des_out


# -------------------------------------------------------
# 3. Основной метод: вызывает нужные подфильтры
# -------------------------------------------------------
def filter_keypoints(
    kps: List[cv2.KeyPoint],
    des: Optional[np.ndarray],
    logger=None,
    **kwargs
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Универсальный интерфейс фильтрации keypoints.
    В зависимости от kwargs вызывает соответствующие фильтры.
    
    Пример:
    --------
    filter_keypoints(kps, des,
        use_grid=True, grid_size=(4,4), max_per_cell=50,
        use_nms=True, nms_radius=8
    )
    """
    if not kps:
        return kps, des
    
    before_count = len(kps)

    use_grid: bool = kwargs.get("use_grid", False)
    use_nms: bool = kwargs.get("use_nms", False)

    logger.info("[FilterKeypoints] Начало фильтрации Keypoints")

    if use_grid:
        grid_size: Tuple[int, int] = kwargs.get("grid_size", (3, 3))
        max_per_cell: int = kwargs.get("max_per_cell", 100)
        kps, des = filter_keypoints_grid(kps, des, grid_size=grid_size, max_per_cell=max_per_cell)
        if logger:
            logger.info(f"[FilterKeypoints] После GRID: {before_count} → {len(kps)}")

    if use_nms:
        nms_radius: int = kwargs.get("nms_radius", 8)
        kps, des = filter_keypoints_nms(kps, des, radius=nms_radius)
        if logger:
            logger.info(f"[FilterKeypoints] После NMS: {before_count} → {len(kps)}")

    if logger:
        logger.info(f"[FilterKeypoints] Финальный результат: {len(kps)} точек.")
    
    logger.info("[FilterKeypoints] Завершение фильтрации Keypoints")
    
    return kps, des

