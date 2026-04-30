import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """加载 .env 文件（优先从 EXE 同目录读取，普通 Python 从项目根目录读取）"""
    try:
        from dotenv import load_dotenv
        if getattr(sys, 'frozen', False):
            env_path = Path(sys.executable).parent / ".env"
        else:
            env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


_load_dotenv()


def _get_data_dir() -> Path:
    if os.getenv("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    if getattr(sys, 'frozen', False):
        # PyInstaller --onefile: __file__ 指向临时解压目录，数据应存在 EXE 同目录
        return Path(sys.executable).parent / "data"
    return Path(__file__).parent.parent / "data"


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
    # YOLO 推理设备: "auto"（自动选 CUDA→CPU）| "cpu" | "cuda" | "0"
    YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "auto")

    # 报警冷却（秒）
    ALERT_COOLDOWN: int = int(os.getenv("ALERT_COOLDOWN", "10"))

    # 帧队列大小
    FRAME_QUEUE_SIZE: int = int(os.getenv("FRAME_QUEUE_SIZE", "2"))

    # MediaMTX WebRTC 基础 URL，浏览器通过此地址拉取 WHEP 流
    MEDIAMTX_WEBRTC_URL: str = os.getenv("MEDIAMTX_WEBRTC_URL", "http://192.168.2.23:8889")

    # 认证
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "luculent1!")
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "ai-detector-secret-change-me")


settings = Settings()
