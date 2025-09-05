
import math

import numpy as np
import cv2
from numba import njit
from scipy.spatial.transform import Rotation


# ------------------------------
# Угловые операции на окружности S1
# ------------------------------

@njit
def s1_diff_deg(angle1: float, angle2: float) -> float:
    """
    Разница между углами (в градусах) на окружности S1.
    Возвращает значение в диапазоне [-180, 180].

    Пример:
        s1_diff_deg(350, 10) -> -20
    """
    diff = (angle1 - angle2) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


@njit
def s1_dist_deg(angle1: float, angle2: float) -> float:
    """
    Положительное расстояние между углами (в градусах) на окружности S1.
    Результат в [0, 180].

    Пример:
        s1_dist_deg(350, 10) -> 20
    """
    diff = (angle1 - angle2) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return abs(diff)


k2pi = 2.0 * math.pi


@njit
def s1_diff_rad(angle1: float, angle2: float) -> float:
    """
    Разница между углами (в радианах) на окружности S1.
    Возвращает значение в диапазоне [-pi, pi].

    Пример:
        s1_diff_rad(6.1, 0.1) -> -0.283...
    """
    diff = (angle1 - angle2) % k2pi
    if diff > math.pi:
        diff -= k2pi
    return diff


@njit
def s1_dist_rad(angle1: float, angle2: float) -> float:
    """
    Положительное расстояние между углами (в радианах) на окружности S1.
    Результат в [0, pi].

    Пример:
        s1_dist_rad(6.1, 0.1) -> 0.283...
    """
    diff = (angle1 - angle2) % k2pi
    if diff > math.pi:
        diff -= k2pi
    return abs(diff)


# ------------------------------
# Переходы между SE(3), Sim(3) и матрицами
# ------------------------------

@njit
def poseRt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Собрать SE(3) матрицу из R (3x3) и t (3,).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@njit
def inv_poseRt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Инвертировать SE(3), заданное R (3x3) и t (3,).
    """
    T = np.eye(4)
    R_T = R.T
    T[:3, :3] = R_T
    T[:3, 3] = -R_T @ np.ascontiguousarray(t)
    return T


@njit
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
# Класс Sim3Pose
# ------------------------------

class Sim3Pose:
    """
    Поза в пространстве Sim(3): (R, t, s).
    Хранит вращение R, перенос t и масштаб s.
    """

    def __init__(self, R=np.eye(3), t=np.zeros(3), s=1.0):
        self.R = np.array(R, dtype=float)
        self.t = np.array(t, dtype=float).reshape(3, 1)
        self.s = float(s)
        assert self.s > 0
        self._T = None
        self._inv_T = None

    def __repr__(self):
        return f"Sim3Pose(R={self.R}, t={self.t.ravel()}, s={self.s})"

    def from_matrix(self, T: np.ndarray):
        """
        Инициализация из 4x4 матрицы Sim(3).
        """
        R = T[:3, :3]
        row_norms = np.linalg.norm(R, axis=1)
        self.s = row_norms.mean()
        self.R = R / self.s
        self.t = T[:3, 3].reshape(3, 1)
        return self

    def from_se3_matrix(self, T: np.ndarray):
        """
        Инициализация из SE(3) (без масштаба).
        """
        self.s = 1.0
        self.R = T[:3, :3]
        self.t = T[:3, 3].reshape(3, 1)
        return self

    def matrix(self) -> np.ndarray:
        """
        Вернуть 4x4 матрицу Sim(3).
        """
        if self._T is None:
            self._T = poseRt(self.R * self.s, self.t.ravel())
        return self._T

    def inverse(self):
        """
        Вернуть обратную позу Sim(3).
        """
        return Sim3Pose(self.R.T,
                        -1.0 / self.s * self.R.T @ self.t,
                        1.0 / self.s)

    def inverse_matrix(self) -> np.ndarray:
        """
        Вернуть обратную матрицу 4x4.
        """
        if self._inv_T is None:
            self._inv_T = np.eye(4)
            sR_inv = 1.0 / self.s * self.R.T
            self._inv_T[:3, :3] = sR_inv
            self._inv_T[:3, 3] = -sR_inv @ self.t.ravel()
        return self._inv_T

    def to_se3_matrix(self) -> np.ndarray:
        """
        Преобразовать в SE(3) матрицу (масштаб вынесен в t).
        """
        return poseRt(self.R, self.t.squeeze() / self.s)

    def copy(self):
        return Sim3Pose(self.R.copy(), self.t.copy(), self.s)

    def map(self, p3d: np.ndarray) -> np.ndarray:
        """
        Отобразить одну 3D-точку.
        """
        return self.s * self.R @ p3d.reshape(3, 1) + self.t

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """
        Отобразить массив 3D-точек (N,3).
        """
        return (self.s * self.R @ points.T + self.t).T

    def __matmul__(self, other):
        """
        Перегрузка оператора @ для композиции поз.
        """
        if isinstance(other, Sim3Pose):
            s_res = self.s * other.s
            R_res = self.R @ other.R
            t_res = self.s * self.R @ other.t + self.t
            return Sim3Pose(R_res, t_res, s_res)
        elif isinstance(other, np.ndarray) and other.shape == (4, 4):
            R_other = other[:3, :3]
            row_norms = np.linalg.norm(R_other, axis=1)
            s_other = row_norms.mean()
            R_other = R_other / s_other
            t_other = other[:3, 3].reshape(3, 1)
            s_res = self.s * s_other
            R_res = self.R @ R_other
            t_res = self.s * self.R @ t_other + self.t
            return Sim3Pose(R_res, t_res, s_res)
        else:
            raise TypeError(f"Unsupported operand for @: {type(other)}")


# ------------------------------
# Нормализация векторов
# ------------------------------

@njit
def normalize_vector(v: np.ndarray):
    """
    Нормализация вектора.
    Возвращает (v_normalized, norm).
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v, norm
    return v / norm, norm


@njit
def normalize_vector2(v: np.ndarray):
    """
    Нормализация вектора.
    Возвращает только нормализованный вектор.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


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
    uv1 = np.ones((pts.shape[0], 3), dtype=pts.dtype)
    uv1[:, 0:2] = pts
    return (Kinv @ uv1.T).T[:, 0:2]


# ------------------------------
# Линейная алгебра
# ------------------------------

@njit
def skew(w: np.ndarray) -> np.ndarray:
    """
    Построить кососимметричную матрицу из вектора w (3,).
    """
    wx, wy, wz = w.ravel()
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]])


@njit
def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Hamming-дистанция между двумя бинарными векторами.
    """
    return np.count_nonzero(a != b)


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hamming-дистанции построчно.
    """
    return np.count_nonzero(a != b, axis=1)


@njit
def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Евклидово расстояние между векторами.
    """
    return np.linalg.norm(a.ravel() - b.ravel())


def l2_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Евклидовы расстояния между массивами (построчно).
    """
    return np.linalg.norm(a - b, axis=-1, keepdims=True)


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


def rpy_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Преобразовать матрицу вращения в углы (roll, pitch, yaw) [рад].
    """
    return Rotation.from_matrix(R).as_euler("xyz", degrees=False)


def euler_from_rotation(R: np.ndarray, order="xyz") -> np.ndarray:
    """
    Преобразовать матрицу вращения в углы Эйлера (по порядку).
    """
    return Rotation.from_matrix(R).as_euler(order, degrees=False)


# ------------------------------
# Другие операции с вращениями
# ------------------------------

@njit
def rodrigues_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Построить матрицу вращения по оси и углу (формула Родригеса).
    """
    K = skew(axis)
    I = np.eye(3)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


@njit
def clip_scalar(x: float, x_min: float, x_max: float) -> float:
    """
    Ограничить скаляр в диапазоне [x_min, x_max].
    """
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    return x


@njit
def get_rotation_from_z_vector(vector: np.ndarray) -> np.ndarray:
    """
    Построить матрицу вращения, поворачивающую ось Z в сторону данного вектора.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.eye(3)

    v = vector / norm
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        if np.dot(z, v) > 0:
            return np.eye(3)
        else:
            return np.array([[-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, 1.0]])

    axis /= axis_norm
    angle = np.arccos(clip_scalar(np.dot(z, v), -1.0, 1.0))
    return rodrigues_rotation_matrix(axis, angle)


@njit
def is_rotation_matrix(R: np.ndarray) -> bool:
    """
    Проверить, является ли R допустимой матрицей вращения.
    """
    Rt = R.T
    should_be_identity = Rt @ R
    I = np.eye(R.shape[0])
    norm_diff = np.linalg.norm(should_be_identity - I)
    det = np.linalg.det(R)
    return norm_diff < 1e-8 and abs(det - 1.0) < 1e-6


def closest_orthogonal_matrix(A: np.ndarray) -> np.ndarray:
    """
    Найти ближайшую ортогональную матрицу (через SVD).
    """
    U, _, Vt = np.linalg.svd(A)
    return U @ Vt


def closest_rotation_matrix(A: np.ndarray) -> np.ndarray:
    """
    Найти ближайшую матрицу вращения (детерминант = +1).
    """
    Q = closest_orthogonal_matrix(A)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


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


def xyzq2Tmat(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """
    Собрать 4x4 матрицу SE(3) из позиции и кватерниона.
    """
    R = qvec2rotmat([qx, qy, qz, qw])
    return np.array([[R[0, 0], R[0, 1], R[0, 2], x],
                     [R[1, 0], R[1, 1], R[1, 2], y],
                     [R[2, 0], R[2, 1], R[2, 2], z],
                     [0, 0, 0, 1]])


@njit
def homography_matrix(img: np.ndarray,
                      roll: float, pitch: float, yaw: float,
                      tx=0.0, ty=0.0, tz=0.0) -> np.ndarray:
    """
    Построить матрицу гомографии H (по Hartley-Zisserman, стр. 327).

    img   : изображение (для получения размеров кадра)
    roll, pitch, yaw : углы камеры в радианах
    tx, ty, tz : перенос в метрах
    """
    d = 1.0
    Rwc = yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll)
    Rcw = Rwc.T
    fx = fy = img.shape[1]
    h, w = img.shape[:2]
    cx, cy = w/2, h/2
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    Kinv = np.array([[1.0/fx, 0.0, -cx/fx],
                     [0.0, 1.0/fy, -cy/fy],
                     [0.0, 0.0, 1.0]])
    t_n = np.array([[0.0, 0.0, tx],
                    [0.0, 0.0, ty],
                    [0.0, 0.0, tz]]) / d
    return K @ (Rcw - t_n) @ Kinv
