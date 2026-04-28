"""HTML 页面路由（Jinja2 模板）"""
import sys
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.auth import check_credentials, create_session_token


def _get_templates_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / "web" / "templates"
    return Path(__file__).parent.parent / "web" / "templates"


templates = Jinja2Templates(directory=str(_get_templates_dir()))

router = APIRouter(tags=["web"])

_COOKIE_NAME = "session"
_COOKIE_MAX_AGE = 7 * 24 * 3600


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": error, "username": ""},
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
):
    if check_credentials(username, password):
        token = create_session_token(username)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            _COOKIE_NAME,
            token,
            max_age=_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        return response
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": "用户名或密码错误", "username": username},
        status_code=401,
    )


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(_COOKIE_NAME)
    return response


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: AsyncSession = Depends(get_db)):
    return await preview_page(request, db)


@router.get("/preview", response_class=HTMLResponse)
async def preview_page(request: Request, db: AsyncSession = Depends(get_db)):
    from app.models.stream_source import StreamSource
    from app.services.stream_service import stream_manager
    sources = (await db.execute(select(StreamSource))).scalars().all()
    items = []
    for src in sources:
        d = src.to_dict()
        state = stream_manager.get_state(src.id)
        d["status"] = state.status if state else "stopped"
        d["fps"] = state.fps if state else 0.0
        items.append(d)
    return templates.TemplateResponse(
        request=request,
        name="preview.html",
        context={"streams": items, "active": "preview"},
    )


@router.get("/streams", response_class=HTMLResponse)
async def streams_page(request: Request, db: AsyncSession = Depends(get_db)):
    from app.models.stream_source import StreamSource
    from app.services.stream_service import stream_manager
    sources = (await db.execute(select(StreamSource))).scalars().all()
    items = []
    for src in sources:
        d = src.to_dict()
        state = stream_manager.get_state(src.id)
        d["status"] = state.status if state else "stopped"
        d["fps"] = state.fps if state else 0.0
        items.append(d)
    return templates.TemplateResponse(
        request=request,
        name="streams.html",
        context={
            "streams": items,
            "active": "streams",
            "default_gateway_url": __import__('app.config', fromlist=['settings']).settings.GATEWAY_URL,
        },
    )


@router.get("/detections", response_class=HTMLResponse)
async def detections_page(request: Request, db: AsyncSession = Depends(get_db)):
    from app.models.stream_source import StreamSource
    from app.models.detection import DetectionConfig
    sources = (await db.execute(select(StreamSource))).scalars().all()
    detections = (await db.execute(select(DetectionConfig))).scalars().all()
    return templates.TemplateResponse(
        request=request,
        name="detections.html",
        context={
            "streams": [s.to_dict() for s in sources],
            "detections": [d.to_dict() for d in detections],
            "active": "detections",
        },
    )


@router.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request, db: AsyncSession = Depends(get_db)):
    from app.models.stream_source import StreamSource
    sources = (await db.execute(select(StreamSource))).scalars().all()
    return templates.TemplateResponse(
        request=request,
        name="alerts.html",
        context={"streams": [s.to_dict() for s in sources], "active": "alerts"},
    )


@router.get("/notify", response_class=HTMLResponse)
async def notify_page(request: Request, db: AsyncSession = Depends(get_db)):
    from app.models.notify_config import NotifyChannel
    channels = (await db.execute(
        select(NotifyChannel).order_by(NotifyChannel.id)
    )).scalars().all()
    return templates.TemplateResponse(
        request=request,
        name="notify.html",
        context={"channels": [ch.to_dict() for ch in channels], "active": "notify"},
    )


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="monitor.html",
        context={"active": "monitor"},
    )
