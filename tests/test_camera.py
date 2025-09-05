import math

import numpy as np
import pytest

from visual_slam.camera.camera import (
    CameraBase, 
    PinholeCamera,
    fov2focal, 
    focal2fov
)
from visual_slam.camera.camera_utils import CameraUtils


def test_fov_focal_conversion():
    width = 1280
    fov = math.radians(90)  # 90°
    fx = fov2focal(fov, width)
    assert pytest.approx(fx, rel=1e-6) == 640.0

    calc_fov = focal2fov(fx, width)
    assert pytest.approx(calc_fov, rel=1e-6) == fov


def test_camera_base_intrinsics():
    cam = CameraBase(width=640, height=480, fx=500, fy=500, cx=320, cy=240)

    K = cam.get_intrinsics()
    assert K.shape == (3, 3)
    assert K[0, 0] == 500
    assert K[1, 1] == 500

    Kinv = cam.get_intrinsics_inv()
    assert np.allclose(K @ Kinv, np.eye(3), atol=1e-8)


def test_is_in_image_and_are_in_image():
    cam = CameraBase(width=640, height=480, fx=500, fy=500, cx=320, cy=240)

    # Точка в центре, z > 0
    assert cam.is_in_image([320, 240], z=1.0)

    # Точка за пределами
    assert not cam.is_in_image([1000, 1000], z=1.0)

    # Тест are_in_image
    uvs = np.array([[320, 240], [1000, 1000]])
    zs = np.array([1.0, 1.0])
    res = cam.are_in_image(uvs, zs)
    assert np.array_equal(res, np.array([True, False]))


def test_pinhole_project_unproject():
    cam = PinholeCamera(width=640, height=480, fx=500, fy=500, cx=320, cy=240)

    # Точка в 3D
    point_3d = np.array([[0, 0, 5]], dtype=np.float64)
    uv, z = cam.project(point_3d)
    assert uv.shape == (1, 2)
    assert pytest.approx(z[0], rel=1e-6) == 5.0

    # Обратная проекция
    uv_coords = uv[0]
    back = cam.unproject_3d(uv_coords[0], uv_coords[1], z[0])
    assert np.allclose(back.ravel(), point_3d.ravel(), atol=1e-6)


def test_camera_utils_unproject_and_backproject():
    fx, fy, cx, cy = 500, 500, 320, 240
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    Kinv = np.linalg.inv(K)

    uvs = np.array([[320, 240], [420, 240]], dtype=float)
    depths = np.array([5.0, 5.0])

    # unproject_points
    norm = CameraUtils.unproject_points(uvs, Kinv)
    norm_nb = CameraUtils.unproject_points_numba(uvs, Kinv)
    assert np.allclose(norm, norm_nb, atol=1e-8)

    # unproject_points_3d
    pts3d = CameraUtils.unproject_points_3d(uvs, depths, Kinv)
    pts3d_nb = CameraUtils.unproject_points_3d_numba(uvs, depths, Kinv)
    assert np.allclose(pts3d, pts3d_nb, atol=1e-8)

    # project
    uv_proj, z = CameraUtils.project(pts3d, K)
    assert uv_proj.shape == (2, 2)
    assert np.allclose(z, depths, atol=1e-8)
