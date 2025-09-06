import os
import json
from typing import Any, Dict, Optional


class Config:
    """
    Класс конфигурации для алгоритма SLAM.
    Хранит все основные параметры (камера, датасет, SLAM-модули).
    """

    def __init__(
        self, 
        config_path: Optional[str] = None
    ):
        
        
        ll = 0
        
        
        
        
        

        if config_path is not None:
            self.load(config_path)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return self.__dict__

    def save(self, path: str):
        """Сохранение конфигурации в JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path: str):
        """Загрузка конфигурации из JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Config] Warning: unknown parameter {key}")

    def __repr__(self):
        return f"Config({json.dumps(self.to_dict(), indent=2)})"
