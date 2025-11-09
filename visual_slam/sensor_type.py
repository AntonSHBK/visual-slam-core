from typing import Optional, Dict


class SensorItem:
    __slots__ = ("index", "name", "description", "params")

    def __init__(
        self,
        index: int,
        name: str,
        description: str = "",
        params: Optional[Dict] = None,
    ):
        self.index = index
        self.name = name
        self.description = description
        self.params = params or {}

    def __repr__(self):
        return f"<SensorItem {self.name} ({self.index}): {self.description}>"

    def __eq__(self, other):
        if isinstance(other, SensorItem):
            return self.index == other.index
        return False

    def __hash__(self):
        return hash(self.index)


class SensorType:
    MONOCULAR = SensorItem(0, "MONOCULAR", "Одиночная камера (без глубины)")
    STEREO = SensorItem(1, "STEREO", "Стерео камера (левая + правая)")
    RGBD = SensorItem(2, "RGBD", "RGB + глубина (например, Realsense, Kinect)")

    @classmethod
    def all(cls):
        return [cls.MONOCULAR, cls.STEREO, cls.RGBD]

    @classmethod
    def by_index(cls, idx: int) -> Optional[SensorItem]:
        for s in cls.all():
            if s.index == idx:
                return s
        return None
