from typing import Dict, Optional, Tuple


class Observations:
    __slots__ = ("_data",)

    def __init__(self):
        self._data: Dict[int, Dict[int, int]] = {}

    def add(self, kf_id: int, cam_id: int, kp_idx: int) -> bool:
        """Добавить наблюдение. Возвращает False, если уже есть."""
        if kf_id not in self._data:
            self._data[kf_id] = {cam_id: kp_idx}
            return True

        cam_dict = self._data[kf_id]

        if cam_id not in cam_dict:
            cam_dict[cam_id] = kp_idx
            return True

        return False
    
    def update(self, kf_id: int, cam_id: int, kp_idx: int):
        """
        Явное обновление наблюдения.
        """
        if kf_id not in self._data:
            self._data[kf_id] = {}

        self._data[kf_id][cam_id] = kp_idx

    def remove(self, kf_id: int, cam_id: Optional[int] = None) -> bool:
        """Удалить наблюдение по keyframe_id."""
        if kf_id not in self._data:
            return False

        if cam_id is None:
            self._data.pop(kf_id, None)
            return True

        if cam_id in self._data[kf_id]:
            del self._data[kf_id][cam_id]

            if not self._data[kf_id]:
                del self._data[kf_id]

            return True

        return False

    def get(self, kf_id: int, cam_id: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """Получить (cam_id, kp_idx) по keyframe_id."""
        if kf_id not in self._data:
            return None

        if cam_id is None:
            return self._data[kf_id]

        if cam_id not in self._data[kf_id]:
            return None

        return (cam_id, self._data[kf_id][cam_id])

    def has(self, kf_id: int, cam_id: Optional[int] = None) -> bool:
        if kf_id not in self._data:
            return False

        if cam_id is None:
            return True

        return cam_id in self._data[kf_id]

    def clear(self):
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data.items())

    def items(self):
        return self._data.items()

    def keyframes(self):
        """Список всех kf_id."""
        return list(self._data.keys())

    def cameras(self, kf_id: int):
        """Список cam_id для данного kf_id."""
        if kf_id not in self._data:
            return []
        return list(self._data[kf_id].keys())

    def keypoints(self, kf_id: int):
        """Список kp_idx по камерам для данного KF."""
        if kf_id not in self._data:
            return []
        return list(self._data[kf_id].values())

    def __repr__(self):
        return f"<Observations kf={len(self._data)}>"
