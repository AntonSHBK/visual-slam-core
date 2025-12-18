import numpy as np

from scipy.spatial.transform import Rotation

from visual_slam.utils.geometry import (
    poseRt,
    inv_T,
    rotmat2qvec,
    qvec2rotmat,
    rpy_from_rotation_matrix,
)

class Pose:
    def __init__(
        self,
        T: np.ndarray = None,
        R: np.ndarray = None,
        t: np.ndarray = None,
    ):
        """
        Можно задать:
        - T (4x4) полную матрицу SE(3)
        - или отдельно R (3x3) и t (3,)
        """
        if T is not None:
            self._T = np.array(T, dtype=float)
        elif R is not None and t is not None:
            self._T = poseRt(np.array(R, dtype=float), np.array(t, dtype=float))
        else:
            self._T = np.eye(4, dtype=float)

    # ------------------------------
    # Представления
    # ------------------------------

    def __repr__(self):
        R, t = self.R, self.t
        return f"Pose(R={R.tolist()}, t={t.tolist()})"

    def __array__(self, dtype=None):
        return np.asarray(self._T, dtype=dtype)
    
    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.as_matrix() @ other.as_matrix())
    
    def __iter__(self):
        return iter(self._T)

    def __getitem__(self, idx):
        return self._T[idx]

    def __len__(self):
        return len(self._T)

    def copy(self):
        return Pose(self._T.copy())

    # ------------------------------
    # Доступ к компонентам
    # ------------------------------

    @property
    def T(self) -> np.ndarray:
        """4x4 матрица SE(3)."""
        return self._T

    @property
    def R(self) -> np.ndarray:
        """3x3 матрица вращения."""
        return self._T[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """3D-вектор переноса (1D, shape=(3,))."""
        return self._T[:3, 3]

    @property
    def quaternion(self) -> np.ndarray:
        """Кватернион [qx, qy, qz, qw]."""
        return rotmat2qvec(self.R)

    @property
    def euler_rad(self) -> np.ndarray:
        """Эйлеровы углы (roll, pitch, yaw) в радианах."""
        return rpy_from_rotation_matrix(self.R, degrees=False)

    @property
    def euler_deg(self) -> np.ndarray:
        """Эйлеровы углы (roll, pitch, yaw) в градусах."""
        return rpy_from_rotation_matrix(self.R, degrees=True)

    # ------------------------------
    # Базовые операции
    # ------------------------------
    
    def set_matrix(self, T: np.ndarray):
        """Полностью заменить матрицу SE(3)."""
        assert T.shape == (4, 4)
        self._T = T.copy()

    def set_RT(self, R: np.ndarray, t: np.ndarray):
        assert R.shape == (3, 3)
        t = np.asarray(t).reshape(3)

        self._T = np.eye(4)
        self._T[:3, :3] = R
        self._T[:3, 3] = t


    def set_translation(self, t: np.ndarray):
        """Изменить только вектор переноса."""
        assert t.shape == (3,)
        self._T[:3, 3] = t

    def set_rotation(self, R: np.ndarray):
        """Изменить только вращение."""
        assert R.shape == (3, 3)
        self._T[:3, :3] = R

    def inverse(self) -> "Pose":
        """Инвертировать позу."""
        return Pose(inv_T(self._T))

    def compose(self, other: "Pose") -> "Pose":
        """Композиция поз: self @ other."""
        return Pose(self._T @ other._T)

    def relative_to(self, ref: "Pose") -> "Pose":
        """Относительная поза текущей к ref (ΔPose = ref⁻¹ @ self)."""
        return Pose(inv_T(ref._T) @ self._T)

    def transform_points(self, pts3d: np.ndarray) -> np.ndarray:
        """
        Применить преобразование к 3D-точкам.
        pts3d: (N,3)
        """
        pts_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
        return (self._T @ pts_h.T).T[:, :3]

    # ------------------------------
    # Конструкторы
    # ------------------------------

    @classmethod
    def from_RT(cls, R: np.ndarray, t: np.ndarray) -> "Pose":
        return cls(R=R, t=t)

    @classmethod
    def from_quaternion(cls, qvec: np.ndarray, t: np.ndarray) -> "Pose":
        R = qvec2rotmat(qvec)
        return cls(R=R, t=t)

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float, t=None, degrees=True) -> "Pose":
        R = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
        if t is None:
            t = np.zeros(3)
        return cls(R=R, t=t)

    # ------------------------------
    # Доп. методы
    # ------------------------------

    def as_matrix(self) -> np.ndarray:
        """Вернуть 4x4 матрицу (копию)."""
        return self._T.copy()

    def as_tuple(self):
        """Вернуть (R, t)."""
        return self.R.copy(), self.t.copy()

    def distance(self, other: "Pose") -> float:
        """Евклидово расстояние между центрами поз."""
        return np.linalg.norm(self.t - other.t)

    def angle(self, other: "Pose", degrees=False) -> float:
        """Угол между ориентациями."""
        R_rel = self.R.T @ other.R
        rot = Rotation.from_matrix(R_rel)
        return np.linalg.norm(rot.as_rotvec(degrees=degrees))