
import math
from typing import Tuple

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation


def poseRt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Собрать SE(3) матрицу из R (3x3) и t (3,).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def inv_poseRt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Инвертировать SE(3), заданное R (3x3) и t (3,).
    """
    T = np.eye(4)
    R_T = R.T
    T[:3, :3] = R_T
    T[:3, 3] = -R_T @ np.ascontiguousarray(t.ravel())
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """
    Инвертировать SE(3), заданное полной матрицей T (4x4).
    """
    R_T = T[:3, :3].T
    t = T[:3, 3]
    ret = np.eye(4)
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ np.ascontiguousarray(t)
    return ret

# ------------------------------
# Нормализация векторов
# ------------------------------

def normalize_vector(v: np.ndarray):
    """
    Нормализация вектора.
    Возвращает (v_normalized, norm).
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v, norm
    return v / norm, norm

# ------------------------------
# Добавление единички для перехода в однородные координаты
# ------------------------------

def add_ones(x: np.ndarray) -> np.ndarray:
    """
    Добавить '1' к точке или массиву точек.
    [u,v] -> [u,v,1]
    [[u,v],...] -> [[u,v,1],...]
    """
    if len(x.shape) == 1:
        return np.array([x[0], x[1], 1.0], dtype=float)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


@njit
def add_ones_numba(uvs: np.ndarray) -> np.ndarray:
    """
    Numba-оптимизированная версия add_ones.
    """
    N = uvs.shape[0]
    out = np.ones((N, 3), dtype=uvs.dtype)
    out[:, 0:2] = uvs
    return out

# ------------------------------
# Однородные координаты
# ------------------------------

@njit
def normalize(Kinv: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Преобразует пиксели в нормализованные координаты.

    pts : (N,2) массив пикселей
    Kinv: (3,3) обратная матрица камеры

    Возвращает: (N,2) массив нормализованных координат [x/z, y/z].
    """
    Kinv64 = Kinv.astype(np.float64)
    pts64 = pts.astype(np.float64)
    uv1 = np.ones((pts64.shape[0], 3), dtype=np.float64)
    uv1[:, 0:2] = pts64
    return (Kinv64 @ uv1.T).T[:, 0:2]


# ------------------------------
# Базовые вращения
# ------------------------------

@njit
def yaw_matrix(yaw: float) -> np.ndarray:
    """Поворот вокруг Z (yaw), угол в радианах."""
    return np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                     [math.sin(yaw),  math.cos(yaw), 0.0],
                     [0.0,            0.0,           1.0]])


@njit
def pitch_matrix(pitch: float) -> np.ndarray:
    """Поворот вокруг Y (pitch), угол в радианах."""
    return np.array([[ math.cos(pitch), 0.0, math.sin(pitch)],
                     [0.0,              1.0, 0.0],
                     [-math.sin(pitch), 0.0, math.cos(pitch)]])


@njit
def roll_matrix(roll: float) -> np.ndarray:
    """Поворот вокруг X (roll), угол в радианах."""
    return np.array([[1.0, 0.0,          0.0],
                     [0.0, math.cos(roll), -math.sin(roll)],
                     [0.0, math.sin(roll),  math.cos(roll)]])


@njit
def rotation_matrix_from_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
    """
    Собрать матрицу вращения из углов yaw, pitch, roll (в градусах).
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rx = roll_matrix(roll)
    Ry = pitch_matrix(pitch)
    Rz = yaw_matrix(yaw)
    return Rz @ Ry @ Rx


def rpy_from_rotation_matrix(R: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Преобразовать матрицу вращения в углы (roll, pitch, yaw) [рад].
    """
    return Rotation.from_matrix(R).as_euler("xyz", degrees=degrees)


def euler_from_rotation(R: np.ndarray, order="xyz", degrees: bool = False) -> np.ndarray:
    """
    Преобразовать матрицу вращения в углы Эйлера (по порядку).
    """
    return Rotation.from_matrix(R).as_euler(order, degrees=degrees)


# ------------------------------
# Кватернионы
# ------------------------------

@njit
def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Преобразовать кватернион [qx,qy,qz,qw] в матрицу вращения.
    """
    qx, qy, qz, qw = qvec
    return np.array([
        [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx*qy - qw*qz), 2.0 * (qx*qz + qw*qy)],
        [2.0 * (qx*qy + qw*qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy*qz - qw*qx)],
        [2.0 * (qx*qz - qw*qy), 2.0 * (qy*qz + qw*qx), 1.0 - 2.0 * (qx**2 + qy**2)]
    ])


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """
    Преобразовать матрицу вращения (3x3) в кватернион [qx,qy,qz,qw].
    Используется метод Шепперда (через матрицу K).
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.ravel()
    K = np.array([
        [Rxx - Ryy - Rzz, Ryx + Rxy, Rzx + Rxz, Ryz - Rzy],
        [Ryx + Rxy, Ryy - Rxx - Rzz, Rzy + Ryz, Rzx - Rxz],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, Rxy - Ryx],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[:, np.argmax(eigvals)]
    if qvec[3] < 0:
        qvec *= -1
    return qvec[[0, 1, 2, 3]]


@njit()
def transform_points_numba(
    points_w: np.ndarray, 
    Rcw: np.ndarray, 
    tcw: np.ndarray
) -> np.ndarray:
    N = points_w.shape[0]
    points_c = np.empty_like(points_w)
    for i in range(N):
        px, py, pz = points_w[i, 0], points_w[i, 1], points_w[i, 2]
        points_c[i, 0] = Rcw[0, 0]*px + Rcw[0, 1]*py + Rcw[0, 2]*pz + tcw[0]
        points_c[i, 1] = Rcw[1, 0]*px + Rcw[1, 1]*py + Rcw[1, 2]*pz + tcw[1]
        points_c[i, 2] = Rcw[2, 0]*px + Rcw[2, 1]*py + Rcw[2, 2]*pz + tcw[2]
    return points_c


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Вычисляет покомпонентную и среднюю репроекционную ошибку.

    Args:
        points_3d (np.ndarray): Массив 3D-точек формы (N, 3) в мировой системе координат.
        points_2d (np.ndarray): Соответствующие 2D-точки (N, 2) в пикселях.
        K (np.ndarray): Матрица внутренних параметров камеры (3×3).
        R (np.ndarray): Матрица вращения (3×3), преобразование из мира в камеру.
        t (np.ndarray): Вектор переноса (3,) для той же системы преобразования.

    Returns:
        Tuple[np.ndarray, float]:
            - errors: вектор ошибок репроекции (N,)
            - mean_error: средняя ошибка (float)
    """

    # 1. Проекция 3D-точек в систему координат камеры
    pts_cam = (R @ points_3d.T + t.reshape(3, 1)).T

    # 2. Переход к нормализованным координатам
    pts_norm = pts_cam[:, :2] / pts_cam[:, 2:].clip(min=1e-8)

    # 3. Применение матрицы камеры для перехода в пиксели
    homog = np.concatenate([pts_norm, np.ones((pts_norm.shape[0], 1))], axis=1)
    proj = (K @ homog.T).T
    proj_2d = proj[:, :2]

    # 4. Вычисление евклидовой ошибки (в пикселях)
    errors = np.linalg.norm(points_2d - proj_2d, axis=1)
    mean_error = float(np.mean(errors))

    return errors, mean_error
