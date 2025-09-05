import cv2
import numpy as np

from visual_slam.camera.camera_utils import(
    CameraUtils, fov2focal, focal2fov
)


class CameraBase:
    def __init__(
        self, 
        width: int, 
        height: int,
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float,
        dist_coeffs=None, 
        fps: int = 30
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
            fps : int
                Частота кадров камеры.
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.fovx = focal2fov(fx, width)
        self.fovy = focal2fov(fy, height)

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5, dtype=float)
        else:
            self.dist_coeffs = np.array(dist_coeffs, dtype=float)

        self.fps = fps

        self.u_min, self.u_max = 0, width
        self.v_min, self.v_max = 0, height

        self.is_distorted = np.linalg.norm(self.dist_coeffs) > 1e-10
        
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
        return CameraUtils.are_in_image_numba(uvs, zs, self.u_min, self.u_max, self.v_min, self.v_max)        

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
    
    def get_render_projection_matrix(self, znear=0.01, zfar=100.0) -> np.ndarray:
        """
        Возвращает матрицу проекции (4x4) для рендеринга в OpenGL/Open3D.
        """
        W, H = self.width, self.height
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy

        left = ((2 * cx - W) / W - 1.0) * W / 2.0
        right = ((2 * cx - W) / W + 1.0) * W / 2.0
        top = ((2 * cy - H) / H + 1.0) * H / 2.0
        bottom = ((2 * cy - H) / H - 1.0) * H / 2.0

        left = znear / fx * left
        right = znear / fx * right
        top = znear / fy * top
        bottom = znear / fy * bottom

        P = np.zeros((4, 4), dtype=float)
        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

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
                f"fps={self.fps}, distorted={self.is_distorted})")


class PinholeCamera(CameraBase):
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
        fps: int = 30, 
        bf: float = None
    ):
        """
        bf : float | None
            baseline * fx (для стереокамеры). Если None — камера моно.
        """
        super().__init__(width, height, fx, fy, cx, cy, dist_coeffs, fps)

        self.K = self.get_intrinsics()
        self.Kinv = self.get_intrinsics_inv()

        self.bf = bf

    # -----------------------------
    # Проекция
    # -----------------------------

    def project(self, points_3d: np.ndarray):
        """
        Проекция 3D точек (N,3) на изображение.

        Возвращает:
            uvs : (N,2) — пиксели
            zs  : (N,)  — глубины
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        return CameraUtils.project_numba(points_3d.astype(np.float64), self.K)

    def project_stereo(self, points_3d: np.ndarray):
        """
        Проекция 3D точек для стереокамеры (N,3).
        Требуется self.bf.

        Возвращает:
            uvs : (N,3) — [u_left, v, u_right]
            zs  : (N,)  — глубины
        """
        if self.bf is None:
            raise ValueError("Stereo projection requires bf (baseline * fx)")
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        return CameraUtils.project_stereo_numba(points_3d.astype(np.float64), self.K, self.bf)

    # -----------------------------
    # Обратная проекция
    # -----------------------------

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
        return CameraUtils.unproject_points_numba(uvs.astype(np.float64), self.Kinv)

    def unproject_points_3d(self, uvs: np.ndarray, depths: np.ndarray):
        """
        Обратная проекция массива пикселей (N,2) + глубины (N,) в 3D (N,3).
        """
        return CameraUtils.unproject_points_3d_numba(
            uvs.astype(np.float64), depths.astype(np.float64), self.Kinv
        )

    # -----------------------------
    # Коррекция искажений
    # -----------------------------

    def undistort_points(self, uvs: np.ndarray):
        """
        Убирает дисторсию у массива пикселей (N,2).
        """
        if self.is_distorted:
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((-1, 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.dist_coeffs, None, self.K)
            return uvs_undistorted.reshape(-1, 2)
        else:
            return uvs

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
                self.K, self.dist_coeffs, None, self.K
            )
            uv_bounds_undistorted = uv_bounds_undistorted.reshape(-1, 2)
        else:
            uv_bounds_undistorted = uv_bounds

        self.u_min = float(np.min(uv_bounds_undistorted[:, 0]))
        self.u_max = float(np.max(uv_bounds_undistorted[:, 0]))
        self.v_min = float(np.min(uv_bounds_undistorted[:, 1]))
        self.v_max = float(np.max(uv_bounds_undistorted[:, 1]))