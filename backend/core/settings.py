import os
from dataclasses import dataclass
from pathlib import Path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_origins(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class RuntimeSettings:
    app_env: str
    app_title: str
    api_host: str
    api_port: int
    model_name: str
    default_performance_mode: bool
    ocr_enabled: bool
    max_objects_per_frame: int
    uploads_dir: Path
    outputs_dir: Path
    db_url: str
    cors_origins: list[str]


def load_settings(base_dir: Path) -> RuntimeSettings:
    db_path = base_dir / os.getenv("DATABASE_PATH", "tracking.db")
    return RuntimeSettings(
        app_env=os.getenv("APP_ENV", "dev"),
        app_title=os.getenv("APP_TITLE", "Object Detection & Tracking API"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_get_int("API_PORT", 8000),
        model_name=os.getenv("MODEL_NAME", "yolov8n.pt"),
        default_performance_mode=_get_bool(
            "DEFAULT_PERFORMANCE_MODE", _get_bool("PERFORMANCE_MODE", False)
        ),
        ocr_enabled=_get_bool("OCR_ENABLED", True),
        max_objects_per_frame=_get_int("MAX_OBJECTS_PER_FRAME", 50),
        uploads_dir=base_dir / os.getenv("UPLOAD_DIR", "uploads"),
        outputs_dir=base_dir / os.getenv("OUTPUT_DIR", "storage/outputs"),
        db_url=os.getenv("DB_URL", f"sqlite:///{str(db_path)}"),
        cors_origins=_get_origins("CORS_ORIGINS", "*"),
    )
