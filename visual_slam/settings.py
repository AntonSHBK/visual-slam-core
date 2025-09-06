import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field

from visual_slam.utils.logging import setup_logging

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Глобальные настройки приложения."""
    
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu", 
        description="Устройство для выполнения моделей: 'cpu' или 'cuda'"
    )
    # Директории
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    CACHE_DIR: Path = Field(default=BASE_DIR / "data" / "cache_dir")
    LOG_DIR: Path = Field(default=BASE_DIR / "logs")

    # Логирование
    LOG_LEVEL: str = Field(default="INFO")
    
    # model_config = SettingsConfigDict(
    #     env_file=BASE_DIR / ".env",
    #     env_file_encoding="utf-8",
    #     case_sensitive=True
    # )

    @field_validator(
        "CACHE_DIR", 
        "LOG_DIR", 
        "DATA_DIR", 
        mode="before"
    )
    @classmethod
    def validate_paths(cls, value: str | Path) -> Path:
        """Автоматически создаёт директории при инициализации."""
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()

# setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)