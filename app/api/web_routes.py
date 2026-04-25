"""HTML 页面路由（Jinja2 模板）"""
import sys
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db


def _get_templates_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / "web" / "templates"
    return Path(__file__).parent.parent / "web" / "templates"


templates = Jinja2Templates(directory=str(_get_templates_dir()))

router = APIRouter(tags=["web"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="streams.html")


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
        context={"streams": items, "active": "streams"},
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


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="monitor.html",
        context={"active": "monitor"},
    )
