import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def _create_logger_handler(
    log_dir: Path,
    log_file: str,
    log_level: str,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    """Создаёт и настраивает file handler — без добавления в логгер."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    return handler

def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO"):
    """Настройка базового логирования и стандартных логгеров."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = log_level.upper()

    # Общий форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        # Консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Общий файл
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # API и Model логгеры
    # _create_logger("api", log_dir, "api.log", log_level, max_bytes=5 * 1024 * 1024, backup_count=3).propagate = True

def get_logger(
    name: str,
    log_dir: Path = Path("logs"),
    log_file: str | None = None,
    log_level: str = "INFO"
):
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())
    
    log_dir = Path(log_dir)

    if log_file:
        log_dir.mkdir(parents=True, exist_ok=True)

        handler_exists = any(
            isinstance(h, RotatingFileHandler)
            and h.baseFilename.endswith(log_file)
            for h in logger.handlers
        )

        if not handler_exists:
            handler = _create_logger_handler(
                log_dir=log_dir,
                log_file=log_file,
                log_level=log_level.upper(),
                max_bytes=5 * 1024 * 1024,
                backup_count=3
            )
            logger.addHandler(handler)

    return logger


