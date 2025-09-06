from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import cv2

try:
    from visual_slam.utils.logging import get_logger
except Exception:
    import logging
    def get_logger(name, log_dir=None, log_file=None, log_level="INFO"):
        logging.basicConfig(level=getattr(logging, log_level))
        return logging.getLogger(name)


class CalibrationBase:
    """
    Базовый интерфейс калибровки.
    Делает: загрузка из файла, проверка, приведение к OpenCV-представлению,
    подготовка undistort/rectify карт и процедур.
    """

    def __init__(self, log_dir="logs"):
        self.logger = get_logger(
            self.__class__.__name__,
            log_dir=log_dir,
            log_file=f"{self.__class__.__name__.lower()}.log",
            log_level="INFO",
        )

    def load_from(self, path: Union[str, Path]):
        """Загрузка калибровки из файла."""
        raise NotImplementedError

    def info(self) -> str:
        """Короткая сводка для логов/консоли."""
        raise NotImplementedError


@dataclass
class MonoCalibration:
    K: np.ndarray
    D: Optional[np.ndarray] = None
    model: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    _map1: Optional[np.ndarray] = field(default=None, repr=False)
    _map2: Optional[np.ndarray] = field(default=None, repr=False)

    def opencv_dist_model(self) -> str:
        if not self.model:
            return "none"
        m = self.model.lower()
        if "equidistant" in m or "fisheye" in m:
            return "fisheye"
        if "radtan" in m or m in ("plumb_bob", "opencv"):
            return "radtan"
        return m

    def init_undistort_maps(self, alpha: float = 0.0, new_size: Optional[Tuple[int, int]] = None):
        if self.image_size is None:
            raise ValueError("image_size не задан — невозможно построить undistort карты.")
        w, h = self.image_size
        new_size = new_size or (w, h)
        dist = self.D if self.D is not None else np.zeros((5, 1), dtype=np.float64)

        if self.opencv_dist_model() == "fisheye":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, dist, (w, h), np.eye(3), balance=alpha
            )
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, dist, np.eye(3), new_K, new_size, cv2.CV_16SC2
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, dist, (w, h), alpha, new_size)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.K, dist, None, new_K, new_size, cv2.CV_16SC2
            )

    def undistort(self, img: np.ndarray) -> np.ndarray:
        if self._map1 is None or self._map2 is None:
            self.init_undistort_maps()
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)


@dataclass
class StereoCalibration:
    # левый/правый intrinsics
    K1: np.ndarray
    D1: Optional[np.ndarray]
    K2: np.ndarray
    D2: Optional[np.ndarray]
    # extrinsics между камерами (R: right<-left, T: right_origin_in_left)
    R: np.ndarray
    T: np.ndarray
    # rectification / projection matrices
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None  # (width, height)

    # карты ремапа
    _lmap1: Optional[np.ndarray] = field(default=None, repr=False)
    _lmap2: Optional[np.ndarray] = field(default=None, repr=False)
    _rmap1: Optional[np.ndarray] = field(default=None, repr=False)
    _rmap2: Optional[np.ndarray] = field(default=None, repr=False)

    def rectify(self, alpha: float = 0.0, zero_disparity: bool = True):
        if self.image_size is None:
            raise ValueError("image_size не задан — невозможно выполнить stereoRectify.")
        size = self.image_size
        D1 = self.D1 if self.D1 is not None else np.zeros((5, 1), dtype=np.float64)
        D2 = self.D2 if self.D2 is not None else np.zeros((5, 1), dtype=np.float64)

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, D1, self.K2, D2, size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0,
            alpha=alpha
        )
        # построим карты
        self._lmap1, self._lmap2 = cv2.initUndistortRectifyMap(
            self.K1, D1, self.R1, self.P1, size, cv2.CV_16SC2
        )
        self._rmap1, self._rmap2 = cv2.initUndistortRectifyMap(
            self.K2, D2, self.R2, self.P2, size, cv2.CV_16SC2
        )

    def rectify_pair(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._lmap1 is None or self._rmap1 is None:
            self.rectify()
        l = cv2.remap(left,  self._lmap1, self._lmap2, cv2.INTER_LINEAR)
        r = cv2.remap(right, self._rmap1, self._rmap2, cv2.INTER_LINEAR)
        return l, r


# ---------------------------------------- #
#      Concrete loader with heuristics     #
# ---------------------------------------- #

class UniversalCalibration(CalibrationBase):
    """
    Универсальная калибровка:
    - Mono: KITTI (.txt), OpenCV/ROS (.yaml/.yml), Kalibr (.yaml camchain)
    - Stereo: KITTI (.txt с P0..P3/Tr_rect), Kalibr stereo chain, ROS Stereo YAML (left/right)
    Возвращает объект MonoCalibration или StereoCalibration.
    """

    def __init__(self, log_dir: str = "logs"):
        super().__init__(log_dir=log_dir)
        self.mono: Optional[MonoCalibration] = None
        self.stereo: Optional[StereoCalibration] = None
        self._source: Optional[Path] = None

    def load_from(self, path: Union[str, Path]) -> "UniversalCalibration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл калибровки не найден: {path.resolve()}")
        self._source = path
        self.logger.info(f"Загружаю калибровку: {path.name}")

        if path.suffix == ".txt":
            self._load_kitti_txt(path)
        elif path.suffix in (".yaml", ".yml"):
            # пробуем по порядку: ROS/OpenCV mono/stereo, затем Kalibr
            if not (self._try_load_ros_yaml(path) or self._try_load_kalibr_yaml(path)):
                raise ValueError(f"Не удалось распознать YAML калибровку: {path}")
        else:
            raise ValueError(f"Неизвестный формат файла калибровки: {path}")

        self.logger.info(self.info())
        return self

    def info(self) -> str:
        lines = []

        if self.mono:
            w, h = (self.mono.image_size or (None, None))
            fx, fy = self.mono.K[0, 0], self.mono.K[1, 1]
            cx, cy = self.mono.K[0, 2], self.mono.K[1, 2]

            lines.extend([
                "",
                "Mono calibration:",
                f"* image_size: {w} x {h}",
                f"* model: {self.mono.model or 'none'}",
                f"* fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}",
                f"* Dist coeffs: {None if self.mono.D is None else len(self.mono.D)}",
                f"* Undistort maps ready: {self.mono._map1 is not None}",
                ""
            ])

        if self.stereo:
            w, h = (self.stereo.image_size or (None, None))
            fx1, fy1 = self.stereo.K1[0, 0], self.stereo.K1[1, 1]
            cx1, cy1 = self.stereo.K1[0, 2], self.stereo.K1[1, 2]
            fx2, fy2 = self.stereo.K2[0, 0], self.stereo.K2[1, 1]
            cx2, cy2 = self.stereo.K2[0, 2], self.stereo.K2[1, 2]

            lines.extend([
                "",
                "Stereo calibration:",
                f"* image_size: {w} x {h}",
                f"* baseline (|T|): {float(np.linalg.norm(self.stereo.T)):.6f} m",
                f"* Q available: {self.stereo.Q is not None}",
                f"* K1: fx={fx1:.2f}, fy={fy1:.2f}, cx={cx1:.2f}, cy={cy1:.2f}",
                f"* K2: fx={fx2:.2f}, fy={fy2:.2f}, cx={cx2:.2f}, cy={cy2:.2f}",
                f"* Dist1: {None if self.stereo.D1 is None else len(self.stereo.D1)} coeffs",
                f"* Dist2: {None if self.stereo.D2 is None else len(self.stereo.D2)} coeffs",
                f"* Rectification computed: {self.stereo.R1 is not None and self.stereo.R2 is not None}",
                ""
            ])

        return "\n".join(lines) if lines else "Пустая калибровка."



    # ---------------- internal loaders ---------------- #

    def _load_kitti_txt(self, path: Path):
        """
        Поддержка KITTI:
        - Mono: 1 строка 12 чисел (P), вернём K=P[:,:3]
        - Stereo: 2 строки по 12 чисел (P0/P1) или с префиксами
        """
        with path.open("r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        def parse_row(prefix: str, lines: list) -> Optional[np.ndarray]:
            for ln in lines:
                parts = ln.split()
                if ln.startswith(prefix):
                    parts = parts[1:]
                try:
                    vals = list(map(float, parts))
                except ValueError:
                    continue
                if len(vals) == 12:
                    return np.array(vals).reshape(3, 4)
            return None

        P0 = parse_row("P0:", lines)
        P1 = parse_row("P1:", lines)

        if P0 is None and len(lines) >= 1:
            vals0 = list(map(float, lines[0].split()))
            if len(vals0) == 12:
                P0 = np.array(vals0).reshape(3, 4)

        if P1 is None and len(lines) >= 2:
            vals1 = list(map(float, lines[1].split()))
            if len(vals1) == 12:
                P1 = np.array(vals1).reshape(3, 4)

        if P0 is not None:
            self.mono = MonoCalibration(K=P0[:, :3], D=None, model="none", image_size=None)
            self.logger.info("Определена моно-калибровка KITTI (из P0).")

        if P0 is not None and P1 is not None:
            K1 = P0[:, :3]
            K2 = P1[:, :3]
            fx = K1[0, 0]
            Tx = -P1[0, 3] / fx
            T = np.array([Tx, 0.0, 0.0], dtype=np.float64).reshape(3, 1)
            R = np.eye(3, dtype=np.float64)

            self.stereo = StereoCalibration(
                K1=K1, D1=None, K2=K2, D2=None, R=R, T=T,
                P1=P0, P2=P1, Q=None, image_size=None
            )
            self.logger.info(f"Определена стерео-калибровка KITTI (baseline={Tx:.3f} м).")

        if self.mono is None and self.stereo is None:
            raise ValueError("KITTI .txt не распознан как mono/stereo.")
    
    def _try_load_ros_yaml(self, path: Path) -> bool:
        """
        Поддержка ROS/OpenCV YAML:
        - Mono: camera_matrix/distortion_coefficients/distortion_model/image_width/height
        - Stereo: отдельные файлы left.yaml/right.yaml или stereo dict
        Эвристика: если в YAML есть поля вида 'left'/'right' — считаем stereo-блоком.
        """
        import yaml
        with path.open('r') as f:
            data = yaml.safe_load(f)

        # stereo как единый yaml
        if isinstance(data, dict) and "left" in data and "right" in data:
            L = data["left"]
            R = data["right"]

            K1 = np.array(L["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
            D1 = np.array(L.get("distortion_coefficients", {}).get("data", []), dtype=np.float64) if L.get("distortion_coefficients") else None
            K2 = np.array(R["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
            D2 = np.array(R.get("distortion_coefficients", {}).get("data", []), dtype=np.float64) if R.get("distortion_coefficients") else None

            im_w = int(L.get("image_width", 0) or 0)
            im_h = int(L.get("image_height", 0) or 0)
            image_size = (im_w, im_h) if im_w and im_h else None

            # extrinsics могут отсутствовать — оставим идентичность; можно позже задать извне
            Rmat = np.eye(3, dtype=np.float64)
            T = np.array([0., 0., 0.], dtype=np.float64).reshape(3, 1)

            self.stereo = StereoCalibration(
                K1=K1, D1=D1, K2=K2, D2=D2, R=Rmat, T=T, image_size=image_size
            )
            self.logger.info("Распознан ROS/OpenCV stereo YAML.")
            return True

        # mono
        if "camera_matrix" in data:
            K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
            dist = np.array(data.get("distortion_coefficients", {}).get("data", []), dtype=np.float64) if data.get("distortion_coefficients") else None
            model = data.get("distortion_model", None)
            im_w = int(data.get("image_width", 0) or 0)
            im_h = int(data.get("image_height", 0) or 0)
            image_size = (im_w, im_h) if im_w and im_h else None

            self.mono = MonoCalibration(K=K, D=dist, model=model, image_size=image_size)
            self.logger.info("Распознан ROS/OpenCV mono YAML.")
            return True

        return False

    def _try_load_kalibr_yaml(self, path: Path) -> bool:
        """
        Поддержка Kalibr (camchain-*.yaml), Euroc-стиль:
        - stereo: два блока с intrinsics, distortion_coeffs, T_cn_cnm1 и т.д.
        - mono: один блок
        """
        import yaml
        with path.open('r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return False

        # Kalibr: верхний уровень — имена камер
        cameras = [k for k, v in data.items() if isinstance(v, dict) and "intrinsics" in v]
        if not cameras:
            return False

        def parse_cam(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], str]:
            fx, fy, cx, cy = d["intrinsics"]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            dist = np.array(d.get("distortion_coeffs", []), dtype=np.float64)
            res = tuple(d.get("resolution", (0, 0)))
            model = d.get("distortion_model", "radtan")
            return K, dist, (res[0], res[1]), model

        if len(cameras) == 1:
            cam = data[cameras[0]]
            K, dist, res, model = parse_cam(cam)
            self.mono = MonoCalibration(K=K, D=dist, model=model, image_size=res if all(res) else None)
            self.logger.info("Распознан Kalibr mono YAML.")
            return True

        if len(cameras) >= 2:
            # считаем cameras[0] — левый, cameras[1] — правый
            L = data[cameras[0]]
            R = data[cameras[1]]
            K1, D1, res1, _ = parse_cam(L)
            K2, D2, res2, _ = parse_cam(R)
            size = res1 if all(res1) else (res2 if all(res2) else None)

            # извлекаем экструinsics T_cn_cnm1 (правой относительно левой или наоборот)
            # Kalibr обычно хранит T_cn_cnm1 в виде 4x4 (R|t)
            def get_RT(dd: Dict[str, Any]) -> Optional[np.ndarray]:
                if "T_cn_cnm1" in dd:
                    return np.array(dd["T_cn_cnm1"], dtype=np.float64).reshape(4, 4)
                return None

            RT = get_RT(R) or get_RT(L)
            if RT is not None:
                Rmat = RT[:3, :3]
                T = RT[:3, 3:4]
            else:
                Rmat = np.eye(3, dtype=np.float64)
                T = np.zeros((3, 1), dtype=np.float64)

            self.stereo = StereoCalibration(
                K1=K1, D1=D1, K2=K2, D2=D2, R=Rmat, T=T, image_size=size
            )
            self.logger.info("Распознан Kalibr stereo YAML.")
            return True

        return False
