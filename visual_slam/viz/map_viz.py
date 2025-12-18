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
        cam_scale = 1.0

        if show_points:
            points, colors = self._get_points()
            if len(points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(pcd)

                cam_scale = self._estimate_scene_scale(points, scale_factor=0.02)
                self.logger.info(
                    f"[show_scene] Добавлено {len(points)} точек карты. "
                    f"Scene scale = {cam_scale:.4f}"
                )
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
            axis = self._get_axis(size=cam_scale * 1.0)
            geometries.append(axis)
            
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

    def _get_camera_geometries(self, keyframes=None, scale=1.0):
        """Создаёт фрустумы камер для KeyFrame."""
        geometries = []
        kfs = self.map.get_keyframes() if keyframes is None else keyframes

        for kf in kfs:
            cam = self._create_camera_frustum(scale=scale * 5)

            T = kf.T_c2w
            cam.transform(T)
            cam.paint_uniform_color([1.0, 0.0, 0.0])

            geometries.append(cam)

        return geometries


    def _create_camera_frustum(self, scale=1.0):
        w = 0.04 * scale
        h = 0.03 * scale
        z = 0.1 * scale

        verts = np.array([
            [0, 0, 0],
            [-w, -h, z],
            [ w, -h, z],
            [ w,  h, z],
            [-w,  h, z],
        ])

        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
        ]

        cam = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(verts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        cam.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

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
        kf_ref: KeyFrame,
        kf_cur: KeyFrame,
        K: np.ndarray,
        window_name: str = "3D Projection Matches",
        max_display: int = None,
        font_scale: float = 0.4,
        wait_key: bool = True,
    ):
        """
        Отображает:
        - keypoints двух ключевых кадров
        - реальные 3D→2D проекции MapPoints на оба кадра
        С учётом новой структуры map_points: Dict[(cam_id, kp_idx), MapPoint]
        """

        # --- Проверяем изображения ---
        if not kf_ref.images or not kf_cur.images:
            self.logger.error("[show_matches] Нет изображений в KeyFrame.")
            return None

        img_ref = kf_ref.images[0].copy()
        img_cur = kf_cur.images[0].copy()

        if img_ref.ndim == 2:
            img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
        if img_cur.ndim == 2:
            img_cur = cv2.cvtColor(img_cur, cv2.COLOR_GRAY2BGR)

        # ============================================================
        # 1. Поиск общих MapPoints
        # ============================================================

        shared_matches = []  
        # формат: [ ((cam_ref, kp_ref_idx), (cam_cur, kp_cur_idx), mp), ... ]

        for (cam_ref, kp_ref_idx), mp in kf_ref.map_points.items():
            if mp is None or not mp.is_valid():
                continue

            # ищем это mp в другом KF
            for (cam_cur, kp_cur_idx), mp2 in kf_cur.map_points.items():
                if mp2 is mp:
                    shared_matches.append(
                        ((cam_ref, kp_ref_idx), (cam_cur, kp_cur_idx), mp)
                    )

        if len(shared_matches) == 0:
            self.logger.warning("[show_matches] Нет общих MapPoints между кадрами.")
            return None

        if max_display is not None:
            shared_matches = shared_matches[:max_display]

        self.logger.info(f"[show_matches] Совпадений: {len(shared_matches)}")

        # ============================================================
        # 2. Параметры для проекции
        # ============================================================
        R_ref = kf_ref.R_w2c
        t_ref = kf_ref.t_w2c.reshape(3, 1)

        R_cur = kf_cur.R_w2c
        t_cur = kf_cur.t_w2c.reshape(3, 1)

        rvec_ref, _ = cv2.Rodrigues(R_ref)
        rvec_cur, _ = cv2.Rodrigues(R_cur)

        img_ref_draw = img_ref.copy()
        img_cur_draw = img_cur.copy()

        # Цвета
        n = len(shared_matches)
        hsv = np.linspace(0, 179, n).astype(np.uint8)
        colors = [
            tuple(map(int,
                      cv2.cvtColor(np.uint8([[[h, 255, 255]]]),
                                   cv2.COLOR_HSV2BGR)[0, 0]
                      ))
            for h in hsv
        ]

        # ============================================================
        # 3. Отрисовка каждого совпадения
        # ============================================================
        for i, ((cam_ref, kp_ref_idx), (cam_cur, kp_cur_idx), mp) in enumerate(shared_matches):
            color = colors[i]

            # 2D keypoints
            kp_ref = kf_ref.keypoints[cam_ref][kp_ref_idx].pt
            kp_cur = kf_cur.keypoints[cam_cur][kp_cur_idx].pt

            kp_ref = tuple(map(int, kp_ref))
            kp_cur = tuple(map(int, kp_cur))

            # 3D точка
            X = mp.position.reshape(1, 3).astype(np.float32)

            # проектируем
            p_ref, _ = cv2.projectPoints(X, rvec_ref, t_ref, K, None)
            p_cur, _ = cv2.projectPoints(X, rvec_cur, t_cur, K, None)

            p_ref = tuple(map(int, p_ref.reshape(2)))
            p_cur = tuple(map(int, p_cur.reshape(2)))

            # рисуем keypoints
            cv2.circle(img_ref_draw, kp_ref, 4, (0, 0, 255), -1)
            cv2.circle(img_cur_draw, kp_cur, 4, (0, 0, 255), -1)

            # перспективные проекции
            cv2.circle(img_ref_draw, p_ref, 4, color, -1)
            cv2.circle(img_cur_draw, p_cur, 4, color, -1)

            cv2.putText(img_ref_draw, str(i), p_ref,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            cv2.putText(img_cur_draw, str(i), p_cur,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        # ============================================================
        # 4. Склейка изображений
        # ============================================================
        h1, w1 = img_ref_draw.shape[:2]
        h2, w2 = img_cur_draw.shape[:2]
        W = max(w1, w2)

        out = np.zeros((h1 + h2, W, 3), dtype=np.uint8)
        out[:h1, :w1] = img_ref_draw
        out[h1:h1 + h2, :w2] = img_cur_draw

        win_name = f"{self.window_name_prefix} - {window_name}"
        cv2.imshow(win_name, out)

        if wait_key:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(win_name)
        else:
            cv2.waitKey(1)

        return out


