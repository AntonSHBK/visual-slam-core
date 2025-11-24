import cv2
import numpy as np

from visual_slam.utils.camera import(
    are_in_image_numba,
    fov2focal, 
    focal2fov,
    project_numba,
    project_stereo_numba,
    unproject_points_numba,
    unproject_points_3d_numba
    
)
from visual_slam.utils.logging import get_logger


class Camera:
    def __init__(
        self,
        width: int, 
        height: int,
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float,
        dist=None, 
        log_dir="logs"
    ):
        """
        Параметры:
            width, height : int
                Размер изображения в пикселях.
            fx, fy : float
                Фокусные расстояния в пикселях.
            cx, cy : float
                Координаты главной точки (principal point).
            dist_coeffs : list | np.ndarray | None
                Коэффициенты дисторсии [k1, k2, p1, p2, k3].
        """
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )
        self.logger.info("Initializing CameraBase with parameters: "
                          f"width={width}, height={height}, fx={fx}, fy={fy}, "
                          f"cx={cx}, cy={cy}, dist_coeffs={dist}")

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.fovx = focal2fov(fx, width)
        self.fovy = focal2fov(fy, height)

        if dist is None:
            self.dist = np.zeros(5, dtype=float)
        else:
            self.dist = np.array(dist, dtype=float)

        self.u_min, self.u_max = 0, width
        self.v_min, self.v_max = 0, height

        self.is_distorted = np.linalg.norm(self.dist) > 1e-10
        
    @property
    def K(self) -> np.ndarray:
        return self.get_intrinsics()
    
    @property
    def Kinv(self) -> np.ndarray:
        return self.get_intrinsics_inv()
        
    def is_in_image(self, uv, z: float = 1.0) -> bool:
        """
        Проверка, что точка находится в пределах изображения и перед камерой.

        uv : (2,) [u,v] — пиксельные координаты
        z  : float      — глубина (по умолчанию 1)

        Возвращает True, если точка в кадре и z > 0.
        """
        return (uv[0] >= self.u_min) and (uv[0] < self.u_max) and \
               (uv[1] >= self.v_min) and (uv[1] < self.v_max) and \
               (z > 0)

    def are_in_image(self, uvs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """
        Векторная версия проверки видимости.

        uvs : (N,2) массив пикселей
        zs  : (N,) массив глубин

        Возвращает булев массив (N,), где True — точка видна в кадре.
        """
        return are_in_image_numba(
            uvs, zs, self.u_min, self.u_max, self.v_min, self.v_max
        )        

    def get_intrinsics(self) -> np.ndarray:
        """
        Возвращает матрицу внутренних параметров (K).
        """
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=float)

    def get_intrinsics_inv(self) -> np.ndarray:
        """
        Возвращает обратную матрицу K^-1.
        """
        return np.linalg.inv(self.get_intrinsics())
    
    def project(self, points_3d: np.ndarray):
        """
        Проекция 3D точек (N,3) на изображение.

        Возвращает:
            uvs : (N,2) — пиксели
            zs  : (N,)  — глубины
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        return project_numba(points_3d.astype(np.float64), self.K)
    
    def unproject(self, uv: np.ndarray):
        """
        Обратная проекция одного пикселя в нормализованную 2D точку (z=1).
        """
        x = (uv[0] - self.cx) / self.fx
        y = (uv[1] - self.cy) / self.fy
        return x, y

    def unproject_3d(self, u: float, v: float, depth: float):
        """
        Обратная проекция одного пикселя (u,v) с глубиной в 3D.
        """
        x = depth * (u - self.cx) / self.fx
        y = depth * (v - self.cy) / self.fy
        return np.array([x, y, depth], dtype=np.float64).reshape(3, 1)

    def unproject_points(self, uvs: np.ndarray):
        """
        Обратная проекция массива пикселей (N,2) в нормализованные координаты (z=1).
        """
        return unproject_points_numba(uvs.astype(np.float64), self.Kinv)

    def unproject_points_3d(self, uvs: np.ndarray, depths: np.ndarray):
        """
        Обратная проекция массива пикселей (N,2) + глубины (N,) в 3D (N,3).
        """
        return unproject_points_3d_numba(
            uvs.astype(np.float64), depths.astype(np.float64), self.Kinv
        )

    def undistort_points(self, uvs: np.ndarray):
        """
        Убирает дисторсию у массива пикселей (N,2).
        """
        if self.is_distorted:
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((-1, 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.dist, None, self.K)
            return uvs_undistorted.reshape(-1, 2)
        else:
            return uvs

        return P

    def set_fovx(self, fovx: float):
        """
        Установить горизонтальный угол обзора (в радианах) и пересчитать fx.
        """
        self.fx = fov2focal(fovx, self.width)
        self.fovx = fovx

    def set_fovy(self, fovy: float):
        """
        Установить вертикальный угол обзора (в радианах) и пересчитать fy.
        """
        self.fy = fov2focal(fovy, self.height)
        self.fovy = fovy

    def __repr__(self):
        return (f"CameraBase(width={self.width}, height={self.height}, "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
                f"distorted={self.is_distorted})")


class PinholeCamera(Camera):
    """
    Камера с моделью pinhole (отсутствие дисторсии или простая коррекция).
    """

    def __init__(
        self, 
        width: int, 
        height: int,
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float,
        dist_coeffs=None, 
        bf: float = None
    ):
        """
        bf : float | None
            baseline * fx (для стереокамеры). Если None — камера моно.
        """
        super().__init__(width, height, fx, fy, cx, cy, dist_coeffs)

        self.bf = bf

    def project_stereo(self, points_3d: np.ndarray):
        """
        Проекция 3D точек для стереокамеры (N,3).

        Возвращает:
            uvs : (N,3) — [u_left, v, u_right]
            zs  : (N,)  — глубины
        """
        if self.bf is None:
            raise ValueError("Stereo projection requires bf (baseline * fx)")
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        return project_stereo_numba(points_3d.astype(np.float64), self.K, self.bf)


    def undistort_image_bounds(self):
        """
        Обновляет границы изображения после коррекции дисторсии.
        """
        uv_bounds = np.array([
            [self.u_min, self.v_min],
            [self.u_min, self.v_max],
            [self.u_max, self.v_min],
            [self.u_max, self.v_max]
        ], dtype=np.float64)

        if self.is_distorted:
            uv_bounds_undistorted = cv2.undistortPoints(
                np.expand_dims(uv_bounds, axis=1),
                self.K, self.dist, None, self.K
            )
            uv_bounds_undistorted = uv_bounds_undistorted.reshape(-1, 2)
        else:
            uv_bounds_undistorted = uv_bounds

        self.u_min = float(np.min(uv_bounds_undistorted[:, 0]))
        self.u_max = float(np.max(uv_bounds_undistorted[:, 0]))
        self.v_min = float(np.min(uv_bounds_undistorted[:, 1]))
        self.v_max = float(np.max(uv_bounds_undistorted[:, 1]))