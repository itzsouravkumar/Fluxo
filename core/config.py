from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "FLUXO"
    debug: bool = False

    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://fluxo:fluxo@localhost:5432/fluxo"

    yolo_model_path: str = "models/yolo11n.pt"
    rl_model_path: str = "models/fluxo_rl_agent_v1.zip"
    tft_model_path: str = "models/fluxo_tft_v1.pt"

    rtsp_default_url: str = ""

    map_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
