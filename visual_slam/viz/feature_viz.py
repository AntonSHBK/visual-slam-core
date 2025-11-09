import cv2
import numpy as np
from typing import List, Optional


class FeatureVisualizer:
    """
    Визуализатор для ключевых точек и матчей с цветовой кодировкой.
    Поддерживает закрытие окна по 'q' и сохранение покадрового анализа.
    """

    def __init__(self, window_name_prefix: str = "SLAM Debug", show: bool = True):
        self.window_name_prefix = window_name_prefix
        self.show = show

    # ==============================================================
    # Визуализация ключевых точек
    # ==============================================================
    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        radius: int = 2,
        window_name: str = "Keypoints",
        wait_key: bool = True,
    ) -> Optional[np.ndarray]:
        if image is None or len(keypoints) == 0:
            return None

        # Преобразуем в BGR, если нужно
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

        # Генерируем уникальные цвета для всех точек (градиент по HSV)
        num_points = len(keypoints)
        hsv_colors = np.linspace(0, 179, num_points, dtype=np.uint8)  # Hue от 0 до 179 (OpenCV HSV диапазон)
        colors = [cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                for h in hsv_colors]

        # Рисуем точки с уникальными цветами
        for kp, c in zip(keypoints, colors):
            pt = tuple(map(int, kp.pt))
            color_bgr = tuple(map(int, c))  # (B, G, R)
            cv2.circle(image, pt, radius, color_bgr, thickness=-1)

        # Визуализация
        if self.show:
            win_name = f"{self.window_name_prefix} - {window_name}"
            cv2.imshow(win_name, image)
            if wait_key:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyWindow(win_name)
            else:
                cv2.waitKey(1)

        return image

    # ==============================================================
    # Визуализация матчей с разными цветами
    # ==============================================================
    def draw_matches(
        self,
        img_ref: np.ndarray,
        img_cur: np.ndarray,
        kps_ref: List[cv2.KeyPoint],
        kps_cur: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        window_name: str = "Matches",
        max_display: Optional[int] = 150,
        wait_key: bool = True,
    ) -> Optional[np.ndarray]:
        if img_ref is None or img_cur is None or matches is None:
            return None

        # Приводим grayscale → BGR
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if len(img_ref.shape) == 2 else img_ref.copy()
        img_cur = cv2.cvtColor(img_cur, cv2.COLOR_GRAY2BGR) if len(img_cur.shape) == 2 else img_cur.copy()

        # Приведение knnMatch → список DMatch
        if len(matches) > 0 and isinstance(matches[0], (list, tuple)):
            matches = [m[0] for m in matches if len(m) > 0]

        matches = matches if max_display is None else matches[:max_display]
        if len(matches) == 0:
            return None

        # --- создаём холст вертикально ---
        h1, w1 = img_ref.shape[:2]
        h2, w2 = img_cur.shape[:2]
        out_img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        out_img[:h1, :w1] = img_ref
        out_img[h1:h1 + h2, :w2] = img_cur

        # --- цветовая палитра (градиент HSV) ---
        num_colors = len(matches)
        hsv_colors = np.linspace(0, 179, num_colors).astype(np.uint8)
        colors = [
            tuple(map(int, cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
            for h in hsv_colors
        ]

        # --- отрисовка матчей ---
        for i, m in enumerate(matches):
            c = colors[i]
            pt1 = tuple(map(int, kps_ref[m.queryIdx].pt))
            pt2 = tuple(map(int, kps_cur[m.trainIdx].pt))

            # смещаем координаты нижнего изображения
            pt2_shifted = (int(pt2[0]), int(pt2[1] + h1))

            # тонкая линия (1 px)
            cv2.line(out_img, pt1, pt2_shifted, c, 1, cv2.LINE_AA)

            # точки — того же цвета, что и линия
            cv2.circle(out_img, pt1, 3, c, -1)
            cv2.circle(out_img, pt2_shifted, 3, c, -1)

        # --- показ изображения ---
        if self.show:
            win_name = f"{self.window_name_prefix} - {window_name}"
            cv2.imshow(win_name, out_img)
            if wait_key:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyWindow(win_name)
            else:
                cv2.waitKey(1)

        return out_img


    # ==============================================================
    # Сетка покрытия фич
    # ==============================================================
    def draw_feature_grid(
        self,
        image: np.ndarray,
        grid_size: tuple = (3, 3),
        color: tuple = (255, 0, 0),
        thickness: int = 1,
        window_name: str = "Feature Coverage",
        wait_key: bool = True,
    ) -> Optional[np.ndarray]:
        if image is None:
            return None

        image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        h, w = image_vis.shape[:2]
        gw, gh = grid_size
        step_x, step_y = w // gw, h // gh

        for i in range(1, gw):
            cv2.line(image_vis, (i * step_x, 0), (i * step_x, h), color, thickness)
        for j in range(1, gh):
            cv2.line(image_vis, (0, j * step_y), (w, j * step_y), color, thickness)

        if self.show:
            win_name = f"{self.window_name_prefix} - {window_name}"
            cv2.imshow(win_name, image_vis)
            if wait_key:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyWindow(win_name)
            else:
                cv2.waitKey(1)

        return image_vis
