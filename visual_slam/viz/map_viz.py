import cv2
import numpy as np
import open3d as o3d
from typing import Optional

from visual_slam.map.map import Map
from visual_slam.map.keyframe import KeyFrame
from visual_slam.utils.logging import get_logger


class MapVisualizer:
    def __init__(
        self,
        map_obj: Map,
        window_name_prefix: str = "MapVisualizer",
        log_dir="logs"
    ):
        """
        Класс для визуализации содержимого карты: фичей, ключевых кадров и 3D точек.

        Parameters
        ----------
        map_obj : Map
            Объект карты, содержащий keyframes и map points.
        window_name_prefix : str
            Префикс имени окна OpenCV (по умолчанию "MapVisualizer").
        """
        self.map = map_obj
        self.window_name_prefix = window_name_prefix
        
        self.logger = get_logger(
            self.__class__.__name__, 
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO"
        )

    def show_keypoints(self, kf: "KeyFrame", wait_key: bool = True):
        """
        Отображает ключевые точки на изображении соответствующего KeyFrame.

        Parameters
        ----------
        kf : KeyFrame
            Ключевой кадр, содержащий keypoints и дескрипторы.
        wait_key : bool, optional
            Если True — ожидать нажатие клавиши "q" для закрытия окна.
        """
        image_vis = kf.image.copy()

        self.logger.info(
            f"[show_keypoints] KeyFrame ID={kf.id}, "
            f"image shape={image_vis.shape}, "
            f"keypoints={len(kf.keypoints)}"
        )

        if not kf.keypoints:
            self.logger.warning("[show_keypoints] Нет ключевых точек для отображения.")
            cv2.imshow(f"{self.window_name_prefix} - KeyFrame {kf.id}", image_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        rng = np.random.default_rng(seed=kf.id)
        colors = rng.integers(0, 255, size=(len(kf.keypoints), 3), dtype=np.uint8)

        for kp, color in zip(kf.keypoints, colors):
            pt = tuple(map(int, kp.pt))
            cv2.circle(image_vis, pt, 2, color.tolist(), -1)

        win_name = f"{self.window_name_prefix} - KeyFrame {kf.id}"
        cv2.imshow(win_name, image_vis)

        self.logger.debug(
            f"[show_keypoints] Использовано {len(colors)} случайных цветов "
            f"(R,G,B)≈{np.mean(colors, axis=0).astype(int).tolist()}"
        )

        if wait_key:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                self.logger.info(f"[show_keypoints] Закрытие окна по клавише 'q' ({win_name}).")
                cv2.destroyWindow(win_name)
        else:
            cv2.waitKey(1)
            
    def show_scene(
        self,
        keyframes: Optional[list] = None,
        show_points: bool = True,
        show_cameras: bool = True,
        show_axes: bool = True,
    ):
        """
        Визуализирует всю карту:
        - 3D точки (MapPoints)
        - положения KeyFrame (камеры)
        - глобальные оси координат
        """
        geometries = []

        if show_points:
            points, colors = self._get_points()
            if len(points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(pcd)
                self.logger.info(f"[show_scene] Добавлено {len(points)} точек карты.")
            else:
                self.logger.warning("[show_scene] Нет точек для отображения.")

        # --- Камеры (только KeyFrame) ---
        if show_cameras:
            kfs = self.map.get_keyframes() if keyframes is None else keyframes
            if len(kfs) == 0:
                self.logger.warning("[show_scene] Нет KeyFrame для визуализации.")
            else:
                cams = self._get_camera_geometries(kfs)
                geometries.extend(cams)
                self.logger.info(f"[show_scene] Добавлено {len(cams)} KeyFrame-фрустумов.")

        # --- Оси координат ---
        if show_axes:
            axis = self._get_axis()
            geometries.append(axis)
            self.logger.info("[show_scene] Добавлены координатные оси.")

        # --- Отображение ---
        if geometries:
            o3d.visualization.draw_geometries(geometries)
        else:
            self.logger.warning("[show_scene] Нет геометрий для отображения.")

    def _get_points(self, keyframes=None):
        points, colors = [], []
        rng = np.random.default_rng(seed=42)
        for mp in self.map.get_points():
            points.append(mp.position)
            if mp.color is not None:
                colors.append(mp.color / 255.0)
            else:
                colors.append(rng.random(3))
        return np.array(points), np.array(colors)

    def _get_camera_geometries(self, keyframes=None):
        """Создаёт пирамидки для поз камер."""
        geometries = []
        kfs = self.map.get_keyframes() if keyframes is None else keyframes
        # scale = self._estimate_scene_scale([kf.pose[:3, 3] for kf in kfs]) * 0.1

        for kf in kfs:
            geom = self._create_camera_frustum()
            T = np.linalg.inv(kf.pose)

            geom.transform(T)
            geom.paint_uniform_color([1, 0, 0])
            geometries.append(geom)

        return geometries


    def _create_camera_frustum(self, scale=1.0):
        """Создаёт простую пирамидку (фрустум) для отображения камеры."""
        w, h, z = 0.5 * scale, 0.3 * scale, scale
        verts = np.array([
            [0, 0, 0],
            [-w, -h, z],
            [w, -h, z],
            [w, h, z],
            [-w, h, z],
        ])
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]
        colors = [[1, 0, 0] for _ in lines]
        cam = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(verts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        cam.colors = o3d.utility.Vector3dVector(colors)
        return cam

    def _get_axis(self, size=1.0):
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    def _estimate_scene_scale(self, points, scale_factor=0.02):
        if len(points) == 0:
            return 1.0
        pts = np.array(points)
        extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return max(0.001, extent * scale_factor)
    
    def show_matches(
        self,
        kf_ref,
        kf_cur,
        pts_3d: np.ndarray,
        K: np.ndarray,
        window_name: str = "3D Projection Matches",
        max_display: int = None,
        font_scale: float = 0.4,
        wait_key: bool = True,
    ):
        """
        Отобразить совпадения и 3D-проекции точек на двух кадрах (расположены вертикально).
        Проекции рассчитываются с учётом реальных поз KeyFrame.
        """
        if not hasattr(kf_ref, "image") or not hasattr(kf_cur, "image"):
            self.logger.error("[show_matches_with_projections] Один из KeyFrame не содержит изображения.")
            return None

        img_ref = kf_ref.image.copy()
        img_cur = kf_cur.image.copy()
        if len(img_ref.shape) == 2:
            img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
        if len(img_cur.shape) == 2:
            img_cur = cv2.cvtColor(img_cur, cv2.COLOR_GRAY2BGR)

        # --- Общие MapPoints между кадрами ---
        shared_matches = []
        for idx_ref, mp_ref in kf_ref.points.items():
            if mp_ref in kf_cur.points.values():
                idx_cur = [i for i, mp_cur in kf_cur.points.items() if mp_cur == mp_ref]
                if idx_cur:
                    shared_matches.append((idx_ref, idx_cur[0]))

        if not shared_matches:
            self.logger.warning("[show_matches_with_projections] Нет общих MapPoints между кадрами.")
            return None

        if max_display:
            shared_matches = shared_matches[:max_display]

        self.logger.info(f"[show_matches_with_projections] Совпадений: {len(shared_matches)}")

        # --- Извлекаем реальные позы ---
        R_ref, t_ref = kf_ref.pose[:3, :3], kf_ref.pose[:3, 3].reshape(3, 1)
        R_cur, t_cur = kf_cur.pose[:3, :3], kf_cur.pose[:3, 3].reshape(3, 1)

        # --- Проецируем 3D-точки с учётом поз ---
        rvec_ref, _ = cv2.Rodrigues(R_ref)
        rvec_cur, _ = cv2.Rodrigues(R_cur)
        proj_ref, _ = cv2.projectPoints(pts_3d, rvec_ref, t_ref, K, None)
        proj_cur, _ = cv2.projectPoints(pts_3d, rvec_cur, t_cur, K, None)
        proj_ref = proj_ref.reshape(-1, 2)
        proj_cur = proj_cur.reshape(-1, 2)

        img_ref_proj = img_ref.copy()
        img_cur_proj = img_cur.copy()

        # --- Цвета для визуализации ---
        num_colors = len(shared_matches)
        hsv_colors = np.linspace(0, 179, num_colors).astype(np.uint8)
        colors = [
            tuple(map(int, cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
            for h in hsv_colors
        ]

        # --- Отрисовка ---
        for i, (idx_ref, idx_cur) in enumerate(shared_matches):
            color = colors[i]
            kp_ref = tuple(map(int, kf_ref.keypoints[idx_ref].pt))
            kp_cur = tuple(map(int, kf_cur.keypoints[idx_cur].pt))

            # реальная проекция 3D-точки в кадре
            if i < len(proj_ref) and i < len(proj_cur):
                p_ref = tuple(map(int, proj_ref[i]))
                p_cur = tuple(map(int, proj_cur[i]))

                # Keypoints (красные кружки)
                cv2.circle(img_ref_proj, kp_ref, 3, (0, 0, 255), -1)
                cv2.circle(img_cur_proj, kp_cur, 3, (0, 0, 255), -1)
                # 3D-проекции (зелёные кружки)
                cv2.circle(img_ref_proj, p_ref, 3, color, -1)
                cv2.circle(img_cur_proj, p_cur, 3, color, -1)
                # подписи
                cv2.putText(img_ref_proj, str(i), p_ref, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
                cv2.putText(img_cur_proj, str(i), p_cur, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # --- Склейка: кадры друг под другом ---
        h1, w1 = img_ref_proj.shape[:2]
        h2, w2 = img_cur_proj.shape[:2]
        W = max(w1, w2)
        out_img = np.zeros((h1 + h2, W, 3), dtype=np.uint8)
        out_img[:h1, :w1] = img_ref_proj
        out_img[h1:h1 + h2, :w2] = img_cur_proj

        win_name = f"{self.window_name_prefix} - {window_name}"
        cv2.imshow(win_name, out_img)
        if wait_key:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(win_name)
        else:
            cv2.waitKey(1)

        return out_img
