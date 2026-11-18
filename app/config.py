from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Weld Defect API"
    debug: bool = False
    yolo_model: Path = Path("models/best.pt")
    yolo_device: str = "cpu"
    yolo_confidence: float = Field(default=0.25, gt=0, le=1)
    yolo_image_size: int = Field(default=640, ge=32, le=4096, multiple_of=32)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
