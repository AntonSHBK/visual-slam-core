from typing import Optional, List, Tuple


class Observation:
    def __init__(
        self,
        kf_idx: int,
        kp_idx: int,
        uv: Optional[Tuple[float, float]] = None,
        scale: Optional[int] = None,
    ):
        """
        kf_idx : int
            Индекс keyframe (например, порядковый номер кадра или ключевого кадра)
        kp_idx : int
            Индекс ключевой точки в этом кадре
        uv : (u,v) | None
            Координаты пикселя на изображении
        scale : int | None
            Уровень пирамиды (если используется)
        """
        self.kf_idx = kf_idx
        self.kp_idx = kp_idx
        self.uv = uv
        self.scale = scale

    def __repr__(self):
        return f"<Obs kf={self.kf_idx}, kp={self.kp_idx}, uv={self.uv}>"
    

class Observations:
    def __init__(self):
        self._items: List[Observation] = []

    def add(self, obs: Observation) -> bool:
        """Добавить наблюдение, если его ещё нет."""
        if not any(o.kf_idx == obs.kf_idx for o in self._items):
            self._items.append(obs)
            return True
        return False

    def remove(self, kf_idx: int):
        """Удалить наблюдение по индексу keyframe."""
        self._items = [o for o in self._items if o.kf_idx != kf_idx]

    def get(self, kf_idx: int) -> Optional[Observation]:
        """Найти наблюдение по индексу keyframe."""
        for o in self._items:
            if o.kf_idx == kf_idx:
                return o
        return None

    def all(self) -> List[Observation]:
        """Вернуть список всех наблюдений."""
        return list(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self):
        return f"<Observations n={len(self)}>"
