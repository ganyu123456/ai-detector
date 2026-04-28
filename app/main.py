"""
AI 检测分析平台 - FastAPI 应用入口
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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

        # 放行登录页及静态资源
        if path == "/login" or path.startswith("/static"):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动 ──────────────────────────────────────────────────
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

app.include_router(streams_router)
app.include_router(detections_router)
app.include_router(alerts_router)
app.include_router(system_router)
app.include_router(notify_router)
app.include_router(web_router)


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
