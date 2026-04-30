"""
AI 检测分析平台 - FastAPI 应用入口
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保项目根目录在 sys.path 中，兼容 `python app/main.py` 和 `python -m app.main` 两种运行方式
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings
from app.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """拦截所有请求，未登录时 Web 页面重定向 /login，API 返回 401。"""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 放行登录页、静态资源及 WebSocket 检测事件流
        if path == "/login" or path.startswith("/static") or path.startswith("/ws/"):
            return await call_next(request)

        # 校验 session cookie
        from app.auth import verify_session_token
        token = request.cookies.get("session")
        user = verify_session_token(token) if token else None

        if user is None:
            if path.startswith("/api"):
                return JSONResponse({"detail": "未登录"}, status_code=401)
            return RedirectResponse(url="/login", status_code=302)

        return await call_next(request)


def _get_static_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / "web" / "static"
    return Path(__file__).parent / "web" / "static"


STATIC_DIR = _get_static_dir()


def _log_hardware_info() -> None:
    """启动时打印硬件与运行时信息，方便排查 GPU 未生效问题。"""
    import platform
    logger.info(f"Platform: {platform.system()} {platform.machine()}")

    # CUDA / GPU
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"PyTorch CUDA: available, GPU={gpu_name}")
        else:
            logger.info("PyTorch CUDA: not available (CPU-only build or no GPU)")
    except ImportError:
        logger.info("PyTorch: not installed")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNXRuntime providers: {providers}")
    except ImportError:
        logger.info("ONNXRuntime: not installed")

    # PyAV codec 探测
    try:
        import av
        from app.services.stream_service import _pick_video_decoder
        decoder, use_hw = _pick_video_decoder()
        hw_codecs = [c for c in av.codec.codecs_available if "cuvid" in c or "vaapi" in c or "videotoolbox" in c or "v4l2" in c]
        logger.info(f"PyAV {av.__version__}: hw_codecs={hw_codecs or 'none'}, selected={decoder} (hw={use_hw})")
    except Exception as e:
        logger.info(f"PyAV probe failed: {e}")

    # YOLO 设备配置
    from app.config import settings
    logger.info(f"YOLO_DEVICE config: {settings.YOLO_DEVICE}")

    # pynvml GPU 状态
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            logger.info(f"GPU {i}: {name}, VRAM={mem.total // 1024 // 1024}MB")
        pynvml.nvmlShutdown()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动 ──────────────────────────────────────────────────
    _log_hardware_info()

    logger.info("Initializing database...")
    await init_db()

    logger.info("Starting RTSP streams...")
    from app.services.stream_service import stream_manager
    await stream_manager.start_all_enabled()

    logger.info("Starting detection workers...")
    from app.services.detection_service import detection_manager
    await detection_manager.start_all_enabled()

    logger.info("AI Detector started")
    yield

    # ── 关闭 ──────────────────────────────────────────────────
    logger.info("Stopping streams...")
    from app.services.stream_service import stream_manager
    for state in stream_manager.all_states():
        await stream_manager.stop(state.stream_id)

    logger.info("AI Detector shutdown complete")


app = FastAPI(
    title="AI 检测分析平台",
    description="从 RTSP 流中进行 YOLO 目标检测、入侵检测、越线检测，支持报警推送",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from app.api.streams import router as streams_router
from app.api.detections import router as detections_router
from app.api.alerts import router as alerts_router
from app.api.system import router as system_router
from app.api.notify import router as notify_router
from app.api.web_routes import router as web_router
from app.api.ws import router as ws_router

app.include_router(streams_router)
app.include_router(detections_router)
app.include_router(alerts_router)
app.include_router(system_router)
app.include_router(notify_router)
app.include_router(web_router)
app.include_router(ws_router)


if __name__ == "__main__":
    import uvicorn
    _reload = False if getattr(sys, 'frozen', False) else settings.DEBUG
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=_reload,
        log_level="info",
    )
