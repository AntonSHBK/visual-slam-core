from typing import Optional, Tuple, List

import numpy as np
import cv2


# ============================================================
# 1. 2D–2D Motion Estimation (Essential Matrix)
# ============================================================

def estimate_motion_from_2d2d(
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    ransac_thresh: float = 0.003,
    prob: float = 0.999,
    logger=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Optional[np.ndarray]]:
    """
    Оценка относительного движения камеры по нормализованным 2D-точкам (через Essential Matrix).
    Используется нормализованная система координат камеры (f=1, pp=(0,0)).

    Параметры
    ----------
    pts_ref : (N,2)
        Точки на предыдущем кадре (в нормализованных координатах).
    pts_cur : (N,2)
        Точки на текущем кадре (в нормализованных координатах).
    ransac_thresh : float, optional
        Порог ошибки репроекции в нормализованных единицах (~0.001–0.01).
    prob : float, optional
        Вероятность успеха RANSAC (по умолчанию 0.999).

    Возвращает
    ----------
    R : (3,3)
        Матрица вращения между кадрами.
    t : (3,1)
        Вектор переноса (нормированный, ||t||=1).
    inliers : int
        Количество инлайеров, найденных RANSAC.
    mask : np.ndarray
        Маска инлайеров (булев массив длиной N).
    """
    if pts_ref.shape[0] < 5 or pts_cur.shape[0] < 5:
        return None, None, 0, None

    ransac_method = getattr(cv2, "USAC_MSAC", cv2.RANSAC)
    E, mask = cv2.findEssentialMat(
        pts_cur, pts_ref,
        focal=1.0, pp=(0.0, 0.0),
        method=ransac_method,
        prob=prob,
        threshold=ransac_thresh,
    )

    if E is None or mask is None:
        return None, None, 0, None

    inliers, R, t, mask_pose = cv2.recoverPose(
        E, pts_cur, pts_ref,
        focal=1.0, pp=(0.0, 0.0), mask=mask
    )

    if inliers is None or inliers < 5:
        return None, None, 0, mask_pose

    return R, t, int(inliers), mask_pose


# ============================================================
# 2. 2D–3D Motion Estimation (PnP)
# ============================================================

def estimate_motion_from_2d3d(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
    use_ransac: bool = True,
    iterations_count: int = 100,
    reprojection_error: float = 3.0,
    confidence: float = 0.99,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[np.ndarray]]:
    """
    Оценка позы камеры по 2D–3D соответствиям (PnP).

    Параметры
    ----------
    pts3d : (N,3)
        Трёхмерные точки карты.
    pts2d : (N,2)
        Соответствующие 2D-точки на изображении.
    K : (3,3)
        Внутренняя матрица камеры.
    dist : np.ndarray | None
        Коэффициенты дисторсии (если есть).
    use_ransac : bool, optional
        Использовать ли RANSAC (по умолчанию True).
    iterations_count : int, optional
        Количество итераций RANSAC (по умолчанию 100).
    reprojection_error : float, optional
        Ошибка репроекции для RANSAC (по умолчанию 3.0).
    confidence : float, optional
        Уровень доверия для RANSAC (по умолчанию 0.99).

    Возвращает
    ----------
    R : (3,3) np.ndarray | None
        Вращение.
    t : (3,1) np.ndarray | None
        Перенос.
    success : bool
        Успешно ли оценена поза.
    inliers : np.ndarray | None
        Индексы инлайеров.
    """
    if pts3d.shape[0] < 4 or pts2d.shape[0] < 4:
        return None, None, False, None

    if dist is None:
        dist = np.zeros(5)

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=K,
            distCoeffs=dist,
            iterationsCount=iterations_count,
            reprojectionError=reprojection_error,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None

    if not success:
        return None, None, False, inliers

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t, True, inliers


# ============================================================
# 3. Триангуляция точек
# ============================================================

def triangulate_points(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Триангуляция 3D-точек из двух наборов 2D-точек (в пикселях).

    Использует внутреннюю матрицу K, то есть точки подаются в пиксельных координатах.

    Parameters
    ----------
    K : np.ndarray (3x3)
        Матрица внутренних параметров камеры.
    R : np.ndarray (3x3)
        Вращение между первой и второй камерой.
    t : np.ndarray (3x1)
        Перенос между первой и второй камерой.
    pts1 : np.ndarray (N,2)
        Точки на первом изображении (в пикселях).
    pts2 : np.ndarray (N,2)
        Точки на втором изображении (в пикселях).

    Returns
    -------
    pts_3d : np.ndarray (N,3)
        Триангулированные 3D-точки.
    mask_valid : np.ndarray (N,)
        Булева маска точек с ненулевой однородной координатой (w != 0).
    """
    # Матрицы проекций
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # Триангуляция
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Маска: w != 0
    mask_valid = np.abs(pts_4d[3]) > 1e-8
    
    # mask_valid = (
    #     (pts_3d[:, 2] > 0) &
    #     # (pts_3d[:, 2] < 100.0) &  # опционально
    #     np.isfinite(pts_3d[:, 2])
    # )

    # Преобразуем в неомогенные координаты
    pts_3d = (pts_4d[:3] / pts_4d[3]).T

    return pts_3d, mask_valid

import numpy as np
import cv2


def triangulate_normalized_points(
    pose_1w: np.ndarray,
    pose_2w: np.ndarray,
    pts_1: np.ndarray,
    pts_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Триангуляция 3D-точек из нормализованных координат и поз камер
    (аналог PySLAM: triangulate_normalized_points).

    Parameters
    ----------
    pose_1w : np.ndarray (4x4)
        Поза первой камеры в мировой системе (T_1w = [R1|t1]).
    pose_2w : np.ndarray (4x4)
        Поза второй камеры в мировой системе (T_2w = [R2|t2]).
    pts_1 : np.ndarray (N,2)
        Нормализованные координаты (x/z, y/z) первой камеры.
    pts_2 : np.ndarray (N,2)
        Нормализованные координаты (x/z, y/z) второй камеры.

    Returns
    -------
    points_3d : np.ndarray (N,3)
        Триангулированные 3D-точки в мировой системе координат.
    mask_valid : np.ndarray (N,)
        Булева маска для точек, у которых w != 0.
    """
    # Матрицы проекций (используем только [R|t] без K)
    P1w = pose_1w[:3, :]
    P2w = pose_2w[:3, :]

    # Триангуляция
    points_4d_hom = cv2.triangulatePoints(P1w, P2w, pts_1.T, pts_2.T)

    # Маска валидных точек (ненулевая однородная координата)
    mask_valid = np.abs(points_4d_hom[3]) > 1e-8

    # Декартовы координаты (x, y, z)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T

    return points_3d, mask_valid


    
# ============================================================
# 4.  Проверка на вырожденность
# ============================================================

def compute_baseline(t: np.ndarray) -> float:
    """
    Вычисление длины базиса (baseline) между кадрами.

    Параметры
    ----------
    t : (3,1) или (3,)
        Вектор переноса между кадрами.

    Возвращает
    ----------
    baseline : float
        Длина базиса ||t|| в метрах (или в единицах камеры).
    """
    return float(np.linalg.norm(t))


def compute_normalize_parallax(
    pts_ref_n: np.ndarray,
    pts_cur_n: np.ndarray,
    pose_ref: np.ndarray,
    pose_cur: np.ndarray,
) -> float:
    """
    Вычисление среднего угла параллакса (в градусах) между нормализованными направлениями лучей
    двух кадров, используя их позы (4x4 SE3).

    Параметры
    ----------
    pts_ref_n : (N,2)
        Точки на предыдущем кадре (в нормализованных координатах).
    pts_cur_n : (N,2)
        Точки на текущем кадре (в нормализованных координатах).
    pose_ref : (4,4)
        Поза эталонного кадра (T_ref: камера→мир или ref→world).
    pose_cur : (4,4)
        Поза текущего кадра (T_cur: камера→мир или cur→world).

    Возвращает
    ----------
    parallax_deg : float
        Медианный угол между направлениями лучей (в градусах).
    """
    if pts_ref_n.shape[0] == 0 or pts_cur_n.shape[0] == 0:
        return 0.0

    # 1. Направления лучей в 3D (z=1)
    v_ref = np.hstack([pts_ref_n, np.ones((len(pts_ref_n), 1))])
    v_cur = np.hstack([pts_cur_n, np.ones((len(pts_cur_n), 1))])

    # 2. Вычисляем относительное вращение между позами
    # ΔR = R_ref^T * R_cur
    R_ref = pose_ref[:3, :3]
    R_cur = pose_cur[:3, :3]
    R_rel = R_ref.T @ R_cur

    # 3. Преобразуем лучи текущего кадра в систему координат предыдущего
    v_cur_in_ref = (R_rel @ v_cur.T).T

    # 4. Нормализация векторов
    v_ref /= np.linalg.norm(v_ref, axis=1, keepdims=True)
    v_cur_in_ref /= np.linalg.norm(v_cur_in_ref, axis=1, keepdims=True)

    # 5. Углы между направлениями
    cos_angles = np.einsum("ij,ij->i", v_ref, v_cur_in_ref)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_rad = np.arccos(cos_angles)
    angles_deg = np.degrees(angles_rad)

    # 6. Возвращаем медиану (устойчивую оценку параллакса)
    return float(np.median(angles_deg))


def filter_by_parallax(
    pts_ref_n: np.ndarray,
    pts_cur_n: np.ndarray,
    pose_ref: np.ndarray,
    pose_cur: np.ndarray,
    min_parallax_deg: float = 1.0
):
    """
    Фильтрация пар точек по минимальному углу параллакса между направлениями лучей
    (в градусах), используя позы (4x4 SE3) двух кадров.

    Параметры
    ----------
    pts_ref_n : (N,2)
        Точки на предыдущем кадре (в нормализованных координатах).
    pts_cur_n : (N,2)
        Точки на текущем кадре (в нормализованных координатах).
    pose_ref : (4,4)
        Поза эталонного кадра (T_ref: камера→мир или ref→world).
    pose_cur : (4,4)
        Поза текущего кадра (T_cur: камера→мир или cur→world).
    min_parallax_deg : float
        Минимальный угол параллакса (в градусах), ниже которого точки считаются вырожденными.

    Возвращает
    ----------
    mask : np.ndarray (bool)
        Маска точек, у которых параллакс >= min_parallax_deg.
    ang : np.ndarray (float)
        Углы параллакса (в градусах) для всех точек.
    """
    if len(pts_ref_n) == 0 or len(pts_cur_n) == 0:
        return np.array([], dtype=bool), np.array([])

    # 1. Преобразуем 2D векторы (x, y) → 3D лучи (x, y, 1)
    v_ref = np.hstack([pts_ref_n, np.ones((len(pts_ref_n), 1))])
    v_cur = np.hstack([pts_cur_n, np.ones((len(pts_cur_n), 1))])

    # 2. Относительное вращение между позами
    R_ref = pose_ref[:3, :3]
    R_cur = pose_cur[:3, :3]
    R_rel = R_ref.T @ R_cur  # относительное вращение cur→ref

    # 3. Преобразуем лучи текущего кадра в систему координат ref
    v_cur_in_ref = (R_rel @ v_cur.T).T

    # 4. Нормализация
    v_ref /= np.linalg.norm(v_ref, axis=1, keepdims=True)
    v_cur_in_ref /= np.linalg.norm(v_cur_in_ref, axis=1, keepdims=True)

    # 5. Углы между направлениями лучей
    cos_ang = np.einsum("ij,ij->i", v_ref, v_cur_in_ref)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_ang))

    # 6. Фильтрация по минимальному углу
    mask = ang >= min_parallax_deg
    return mask, ang


def compute_parallax(R: np.ndarray) -> float:
    """
    Вычисление угла параллакса (в градусах) из матрицы вращения.

    Параметры
    ----------
    R : (3,3)
        Матрица вращения между кадрами.

    Возвращает
    ----------
    parallax_deg : float
        Угол поворота (параллакс) в градусах.
    """
    # Через формулу Родригеса: cos(angle) = (trace(R) - 1) / 2
    angle_rad = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def check_feature_coverage(
    kps_ref: List[cv2.KeyPoint],
    kps_cur: List[cv2.KeyPoint],
    shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (3, 3),
    min_per_cell: int = 5,
    min_coverage_ratio: float = 0.6
) -> bool:
    """
    Проверяет равномерность распределения фич по изображению (аналог ImageGrid из PySLAM).

    Parameters
    ----------
    kps_ref : list of cv2.KeyPoint
        Ключевые точки первого кадра.
    kps_cur : list of cv2.KeyPoint
        Ключевые точки второго кадра.
    shape : tuple of int
        Размер изображения (h, w).
    grid_size : tuple of int, optional
        Размер сетки (по умолчанию 3x3).
    min_per_cell : int, optional
        Минимальное количество фич в ячейке, чтобы считать её заполненной.
    min_coverage_ratio : float, optional
        Минимальная доля заполненных ячеек (в диапазоне 0–1), при которой
        изображение считается хорошо покрытым.

    Returns
    -------
    bool
        True — если покрытие изображения фичами достаточно равномерное,
        False — если фичи сосредоточены в малой области (инициализация рискованна).
    """
    h, w = shape[:2]
    gw, gh = grid_size
    cell_w, cell_h = w // gw, h // gh
    coverage = np.zeros((gh, gw), dtype=int)

    for kp in kps_ref + kps_cur:
        x, y = int(kp.pt[0] // cell_w), int(kp.pt[1] // cell_h)
        x = min(x, gw - 1)
        y = min(y, gh - 1)
        coverage[y, x] += 1

    filled = np.sum(coverage > min_per_cell)
    ratio = filled / (gw * gh)

    return ratio >= min_coverage_ratio


def normalize_depth_scale(points_3d: np.ndarray, pose_cur: np.ndarray):
    """
    Масштабирует 3D-точки и позу текущего кадра так, чтобы медианная глубина = 1.0.

    Параметры
    ----------
    points_3d : (N,3)
        Триангулированные 3D-точки (в системе ref-кадра).
    pose_ref : (4,4)
        Поза эталонного кадра (обычно np.eye(4)).
    pose_cur : (4,4)
        Поза текущего кадра (T_cur: камера→мир или cur→ref, в зависимости от системы).

    Возвращает
    ----------
    points_3d_scaled : np.ndarray (N,3)
        Масштабированные 3D-точки.
    pose_cur_scaled : np.ndarray (4,4)
        Масштабированная поза текущего кадра.
    """
    if points_3d.shape[0] == 0:
        return points_3d, pose_cur

    # 1. Медианная глубина
    median_depth = np.median(points_3d[:, 2])

    if median_depth > 1e-6:
        scale = 1.0 / median_depth
        # 2. Масштабируем точки
        points_3d_scaled = points_3d * scale
        # 3. Масштабируем трансляцию текущего кадра
        pose_cur_scaled = pose_cur.copy()
        pose_cur_scaled[:3, 3] *= scale
        return points_3d_scaled, pose_cur_scaled
    else:
        return points_3d, pose_cur


def triangulate_stereo_points(
    kps_left: np.ndarray,
    kps_right: np.ndarray,
    K: np.ndarray,
    baseline: float
) -> np.ndarray:
    """
    Преобразует координаты стерео-точек в 3D с использованием базиса и параметров камеры.

    Parameters
    ----------
    kps_left, kps_right : np.ndarray (N, 2)
        Координаты совпавших точек (x, y) на левом и правом изображениях.
    K : np.ndarray
        Матрица внутренних параметров камеры.
    baseline : float
        Расстояние между камерами (в метрах).

    Returns
    -------
    pts_3d : np.ndarray (N, 3)
        Трёхмерные точки в системе координат левой камеры.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    disparity = kps_left[:, 0] - kps_right[:, 0]
    valid = disparity > 1e-6

    Z = np.zeros_like(disparity)
    Z[valid] = fx * baseline / disparity[valid]
    X = (kps_left[:, 0] - cx) * Z / fx
    Y = (kps_left[:, 1] - cy) * Z / fy

    pts_3d = np.vstack((X, Y, Z)).T
    return pts_3d[valid]

def image_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)