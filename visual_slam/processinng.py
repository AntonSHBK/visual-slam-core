from pathlib import Path
from typing import Union, Optional
import time

from visual_slam.config import Config
from visual_slam.slam import SLAM
from visual_slam.camera import PinholeCamera
from visual_slam.calibration import UniversalCalibration
from visual_slam.source import DatasetSource

class Processing:
    def __init__(
        self,
        video_path: Union[str, Path],
        calibration_file: Union[str, Path],
        config: Config,
        sleep_time: Optional[float] = None,
    ):

        self.config = config
        self.sleep_time = sleep_time

        self.video_source = DatasetSource(video_path)

        self.calibration = UniversalCalibration().load_from(calibration_file)
        K = self.calibration.mono.K

        h, w, _ = self.video_source.get_frame_shape()

        self.camera = PinholeCamera(
            width=w,
            height=h,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            dist_coeffs=self.calibration.mono.D,
        )

        self.slam = SLAM(
            camera=self.camera,
            config=self.config,
        )

    def run(self, max_cycles: Optional[int] = None):
        print("=== Запуск SLAM ===")

        total_frames = self.video_source.num_frames()
        print(f"Всего кадров: {total_frames}")

        count = 0

        while self.video_source.is_ok():
            img, timestamp = self.video_source.get_frame()
            if img is None:
                break

            result = self.slam.track([img], timestamp)
            count += 1

            print(
                f"#{count}  Timestamp={timestamp:.3f}  Result={result}"
            )

            if max_cycles is not None and count >= max_cycles:
                print(f"Остановлено: достигнуто {max_cycles} итераций.")
                break

            if self.sleep_time is not None:
                time.sleep(self.sleep_time)

        self.slam.shutdown()
        print("=== SLAM завершён ===")
