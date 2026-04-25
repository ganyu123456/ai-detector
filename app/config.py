import os
from pathlib import Path


def _get_data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data")))


DATA_DIR = _get_data_dir()
PICTURES_DIR = DATA_DIR / "pictures"
MODELS_DIR = DATA_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
PICTURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8090"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    DB_URL: str = f"sqlite+aiosqlite:///{DATA_DIR / 'detector.db'}"

    DATA_DIR: Path = DATA_DIR
    PICTURES_DIR: Path = PICTURES_DIR
    MODELS_DIR: Path = MODELS_DIR

    # miloco-camera 网关地址（用于自动同步流源）
    GATEWAY_URL: str = os.getenv("GATEWAY_URL", "http://localhost:8080")

    # SMTP 邮件
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.qq.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_TO: str = os.getenv("SMTP_TO", "")
    SMTP_ENABLED: bool = os.getenv("SMTP_ENABLED", "false").lower() == "true"

    # HTTP Webhook
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")

    # YOLO 默认参数
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolo11n.pt")
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))

    # 报警冷却（秒）
    ALERT_COOLDOWN: int = int(os.getenv("ALERT_COOLDOWN", "10"))

    # 帧队列大小
    FRAME_QUEUE_SIZE: int = int(os.getenv("FRAME_QUEUE_SIZE", "2"))


settings = Settings()
