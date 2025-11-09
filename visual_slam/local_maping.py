import threading
import queue

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visual_slam.slam import SLAM
    from visual_slam.map.keyframe import KeyFrame


class LocalMapping(threading.Thread):
    """
    Бэкэнд SLAM: локальное построение карты и bundle adjustment.

    Работает в отдельном потоке: принимает новые keyframe'ы от Tracking,
    добавляет точки в карту, оптимизирует локальное окно и очищает плохие точки.
    """

    def __init__(self, slam, max_queue_size=10):
        super().__init__()
        self.slam = slam
        self.map = slam.map
        self.daemon = True

        # Очередь ключевых кадров для обработки
        self.new_keyframes = queue.Queue(maxsize=max_queue_size)

        # Флаги
        self.running = True
        self.is_idle = True

    def insert_keyframe(self, kf: KeyFrame):
        """
        Поместить новый keyframe в очередь для обработки.
        """
        try:
            self.new_keyframes.put_nowait(kf)
        except queue.Full:
            # Если очередь переполнена, можно дропать или заменять
            print("[LocalMapping] Очередь переполнена, keyframe потерян")

    def run(self):
        """
        Основной цикл LocalMapping: обрабатываем новые keyframe'ы.
        """
        while self.running:
            try:
                kf = self.new_keyframes.get(timeout=0.1)
            except queue.Empty:
                continue

            self.is_idle = False
            self.process_new_keyframe(kf)
            self.is_idle = True

    def process_new_keyframe(self, kf: KeyFrame):
        """
        Обработка нового keyframe: добавляем в карту, строим связи, делаем BA.
        """
        print(f"[LocalMapping] Обработка keyframe {kf.id}")
        self.map.add_keyframe(kf)

        # TODO: Триангуляция новых точек из соседних keyframe'ов
        # TODO: Обновление связей (covisibility graph)
        # TODO: Локальный bundle adjustment
        # TODO: Удаление "плохих" MapPoint'ов

    def stop(self):
        self.running = False
