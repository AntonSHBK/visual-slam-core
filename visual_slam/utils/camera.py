import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from visual_slam.utils.geometry import add_ones, add_ones_numba


def fov2focal(fov_rad: float, pixels: int) -> float:
    """Перевод угла обзора (FOV, рад) в фокусное расстояние (пиксели)."""
    return pixels / (2.0 * math.tan(fov_rad / 2.0))


def focal2fov(focal_px: float, pixels: int) -> float:
    """Перевод фокусного расстояния (пиксели) в угол обзора (FOV, рад)."""
    return 2.0 * math.atan(pixels / (2.0 * focal_px))


def backproject_3d(
    uv: NDArray[np.float64],
    depth: NDArray[np.float64],
    K: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Обратная проекция из 2D пикселей в 3D точки (Python-версия).

    Параметры:
        uv    : массив формы (N,2) — пиксельные координаты [u,v]
        depth : массив формы (N,) — глубины для каждой точки
        K     : матрица внутренних параметров камеры (3x3)

    Возвращает:
        xyz   : массив формы (N,3) — 3D координаты в системе камеры
    """
    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)  # [u,v,1]
    p3d = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
    return p3d.reshape(-1, 3)

@njit
def backproject_3d_numba(
    uv: NDArray[np.float64],
    depth: NDArray[np.float64],
    Kinv: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Ускоренная версия обратной проекции с использованием заранее посчитанной Kinv.
    """
    N = uv.shape[0]
    uv1 = np.ones((N, 3), dtype=np.float64)
    uv1[:, 0:2] = uv
    p3d = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        p = Kinv @ uv1[i]
        p3d[i, :] = depth[i] * p
    return p3d

def project(
    xcs: NDArray[np.float64],
    K: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Проекция 3D точек на изображение (Python-версия).

    Параметры:
        xcs : массив формы (N,3) — 3D точки в системе камеры
        K   : матрица внутренних параметров камеры (3x3)

    Возвращает:
        uv  : массив (N,2) — координаты пикселей
        z   : массив (N,)  — глубины (z-координаты)
    """
    projs = K @ xcs.T
    zs = projs[-1]
    projs = projs[:2] / zs
    return projs.T, zs

@njit
def project_numba(
    xcs: NDArray[np.float64],
    K: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Ускоренная версия проекции (использует numba).
    """
    N = xcs.shape[0]
    projs = K @ xcs.T
    zs = projs[2, :]
    u = projs[0, :] / zs
    v = projs[1, :] / zs
    uv = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        uv[i, 0] = u[i]
        uv[i, 1] = v[i]
    return uv, zs

def project_stereo(
    xcs: NDArray[np.float64],
    K: NDArray[np.float64],
    bf: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Проекция для стереокамеры (Python-версия).

    Параметры:
        xcs : массив (N,3) — 3D точки в системе камеры
        K   : матрица (3x3) внутренних параметров
        bf  : baseline * fx (произведение базы стерео и фокусного расстояния)

    Возвращает:
        uv_stereo : массив (N,3) — координаты [u_left, v, u_right]
        z         : массив (N,)  — глубины
    """
    projs = K @ xcs.T
    zs = projs[-1]
    projs = projs[:2] / zs
    ur = projs[0] - bf / zs
    projs = np.concatenate((projs.T, ur[:, np.newaxis]), axis=1)
    return projs, zs

@njit
def project_stereo_numba(
    xcs: NDArray[np.float64],
    K: NDArray[np.float64],
    bf: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Ускоренная версия стерео-проекции.
    """
    N = xcs.shape[0]
    projs = K @ xcs.T
    zs = projs[2, :]
    u = projs[0, :] / zs
    v = projs[1, :] / zs
    ur = u - bf / zs
    out = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        out[i, 0] = u[i]
        out[i, 1] = v[i]
        out[i, 2] = ur[i]
    return out, zs

def unproject_points(
    uvs: NDArray[np.float64],
    Kinv: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Перевод пикселей в нормализованные координаты (на плоскости z=1).

    Параметры:
        uvs  : массив (N,2) — пиксели
        Kinv : обратная матрица калибровки (3x3)

    Возвращает:
        нормализованные 2D координаты (N,2)
    """
    uv1 = add_ones(uvs.astype(np.float64))
    return (Kinv @ uv1.T).T[:, 0:2]

@njit
def unproject_points_numba(
    uvs: NDArray[np.float64],
    Kinv: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Ускоренная версия нормализации пикселей.
    """
    uv1 = add_ones_numba(uvs)
    out = np.empty((uvs.shape[0], 2), dtype=np.float64)
    for i in range(uvs.shape[0]):
        p = Kinv @ uv1[i]
        out[i, 0] = p[0]
        out[i, 1] = p[1]
    return out

def unproject_points_3d(
    uvs: NDArray[np.float64],
    depths: NDArray[np.float64],
    Kinv: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Перевод пикселей + глубины в 3D точки (Python-версия).

    Параметры:
        uvs    : массив (N,2) — пиксели
        depths : массив (N,)  — глубины
        Kinv   : обратная матрица калибровки (3x3)

    Возвращает:
        3D точки (N,3)
    """
    uv1 = add_ones(uvs.astype(np.float64))
    return (Kinv @ (uv1.T * depths)).T[:, 0:3]

@njit
def unproject_points_3d_numba(
    uvs: NDArray[np.float64],
    depths: NDArray[np.float64],
    Kinv: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Ускоренная версия перевода пикселей + глубин в 3D.
    """
    uv1 = add_ones_numba(uvs)
    out = np.empty((uvs.shape[0], 3), dtype=np.float64)
    for i in range(uvs.shape[0]):
        p = Kinv @ (uv1[i] * depths[i])
        out[i, 0] = p[0]
        out[i, 1] = p[1]
        out[i, 2] = p[2]
    return out

@njit
def are_in_image_numba(
    uvs: NDArray[np.float64],
    zs: NDArray[np.float64],
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float
) -> NDArray[np.bool_]:
    """
    Проверка, находятся ли пиксели в пределах изображения и перед камерой.

    Параметры:
        uvs   : массив (N,2) — пиксели
        zs    : массив (N,)  — глубины
        u_min, u_max, v_min, v_max : границы изображения

    Возвращает:
        булев массив (N,) — True, если точка видна в кадре
    """
    N = uvs.shape[0]
    out = np.empty(N, dtype=np.bool_)
    for i in range(N):
        out[i] = (uvs[i, 0] >= u_min) & (uvs[i, 0] < u_max) & \
                    (uvs[i, 1] >= v_min) & (uvs[i, 1] < v_max) & \
                    (zs[i] > 0)
    return out
