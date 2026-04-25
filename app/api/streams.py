"""流源管理 API：CRUD + 连通性测试 + MJPEG 预览 + 从网关自动同步"""
import asyncio
from typing import Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.stream_source import StreamSource
from app.services.stream_service import stream_manager

router = APIRouter(prefix="/api/streams", tags=["streams"])


class StreamCreate(BaseModel):
    name: str
    rtsp_url: str
    gateway_camera_id: Optional[int] = None
    enabled: bool = True
    description: Optional[str] = None


class StreamUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None


@router.get("")
async def list_streams(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(StreamSource))
    sources = result.scalars().all()
    items = []
    for src in sources:
        d = src.to_dict()
        state = stream_manager.get_state(src.id)
        d["status"] = state.status if state else "stopped"
        d["fps"] = state.fps if state else 0.0
        d["error_msg"] = state.error_msg if state else ""
        items.append(d)
    return {"streams": items}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_stream(body: StreamCreate, db: AsyncSession = Depends(get_db)):
    src = StreamSource(**body.model_dump())
    db.add(src)
    await db.commit()
    await db.refresh(src)

    if src.enabled:
        stream_manager.register(src.id, src.name, src.rtsp_url)
        asyncio.create_task(stream_manager.start(src.id))

    return src.to_dict()


@router.get("/{stream_id}")
async def get_stream(stream_id: int, db: AsyncSession = Depends(get_db)):
    src = await db.get(StreamSource, stream_id)
    if not src:
        raise HTTPException(404, "Stream not found")
    d = src.to_dict()
    state = stream_manager.get_state(stream_id)
    d["status"] = state.status if state else "stopped"
    d["fps"] = state.fps if state else 0.0
    return d


@router.patch("/{stream_id}")
async def update_stream(stream_id: int, body: StreamUpdate, db: AsyncSession = Depends(get_db)):
    src = await db.get(StreamSource, stream_id)
    if not src:
        raise HTTPException(404, "Stream not found")

    update_data = body.model_dump(exclude_none=True)
    for k, v in update_data.items():
        setattr(src, k, v)
    await db.commit()
    await db.refresh(src)

    state = stream_manager.get_state(stream_id)
    if "enabled" in update_data:
        if src.enabled:
            if state is None:
                stream_manager.register(src.id, src.name, src.rtsp_url)
            asyncio.create_task(stream_manager.start(stream_id))
        else:
            asyncio.create_task(stream_manager.stop(stream_id))

    return src.to_dict()


@router.delete("/{stream_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stream(stream_id: int, db: AsyncSession = Depends(get_db)):
    src = await db.get(StreamSource, stream_id)
    if not src:
        raise HTTPException(404, "Stream not found")
    await stream_manager.stop(stream_id)
    stream_manager.unregister(stream_id)
    await db.delete(src)
    await db.commit()


@router.post("/{stream_id}/start")
async def start_stream(stream_id: int, db: AsyncSession = Depends(get_db)):
    src = await db.get(StreamSource, stream_id)
    if not src:
        raise HTTPException(404, "Stream not found")
    if not stream_manager.get_state(stream_id):
        stream_manager.register(src.id, src.name, src.rtsp_url)
    ok = await stream_manager.start(stream_id)
    state = stream_manager.get_state(stream_id)
    return {"ok": ok, "status": state.status if state else "error"}


@router.post("/{stream_id}/stop")
async def stop_stream(stream_id: int):
    await stream_manager.stop(stream_id)
    return {"ok": True, "status": "stopped"}


@router.get("/{stream_id}/test")
async def test_stream_connectivity(stream_id: int, db: AsyncSession = Depends(get_db)):
    """测试 RTSP 流连通性"""
    src = await db.get(StreamSource, stream_id)
    if not src:
        raise HTTPException(404, "Stream not found")

    def _check():
        try:
            import av
            container = av.open(
                src.rtsp_url,
                options={"rtsp_transport": "tcp", "stimeout": "3000000"},
            )
            for packet in container.demux(video=0):
                container.close()
                return True, "连通正常"
        except Exception as e:
            return False, str(e)
        return False, "无帧数据"

    loop = asyncio.get_running_loop()
    ok, msg = await loop.run_in_executor(None, _check)
    return {"ok": ok, "message": msg, "rtsp_url": src.rtsp_url}


@router.get("/{stream_id}/snapshot")
async def get_snapshot(stream_id: int):
    """获取最新一帧 JPEG 快照"""
    state = stream_manager.get_state(stream_id)
    if not state:
        raise HTTPException(404, "Stream not found")
    if not state.latest_frame:
        raise HTTPException(503, "No frame available yet")
    return Response(content=state.latest_frame, media_type="image/jpeg")


@router.get("/{stream_id}/mjpeg")
async def mjpeg_stream(stream_id: int):
    """MJPEG 实时预览流（浏览器直接播放）"""
    state = stream_manager.get_state(stream_id)
    if not state:
        raise HTTPException(404, "Stream not found")

    async def _generator():
        last_frame: Optional[bytes] = None
        idle_ticks = 0
        while True:
            frame = state.latest_frame
            if frame and frame is not last_frame:
                last_frame = frame
                idle_ticks = 0
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            else:
                idle_ticks += 1
                # 流停止超过 10 秒则结束推流
                if state.status not in ("running", "starting") and idle_ticks > 200:
                    break
            await asyncio.sleep(0.05)  # 最高 20 fps

    return StreamingResponse(
        _generator(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


class GatewaySyncBody(BaseModel):
    gateway_url: Optional[str] = None  # 不填则使用环境变量默认值


@router.post("/sync/gateway")
async def sync_from_gateway(
    body: GatewaySyncBody = GatewaySyncBody(),
    db: AsyncSession = Depends(get_db),
):
    """从 miloco-camera 网关自动同步摄像头流源，支持自定义网关地址"""
    gateway_url = (body.gateway_url or settings.GATEWAY_URL).rstrip("/")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{gateway_url}/api/streams",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(502, f"Gateway returned {resp.status}")
                data = await resp.json()
    except aiohttp.ClientError as e:
        raise HTTPException(502, f"Cannot reach gateway: {e}")

    streams = data.get("streams", [])
    added, skipped = 0, 0

    for s in streams:
        rtsp_url = s.get("rtsp_url")
        if not rtsp_url:
            skipped += 1
            continue

        existing = (await db.execute(
            select(StreamSource).where(StreamSource.rtsp_url == rtsp_url)
        )).scalar_one_or_none()

        if existing:
            skipped += 1
            continue

        src = StreamSource(
            name=s.get("name", f"Camera {s.get('camera_id', '')}"),
            rtsp_url=rtsp_url,
            gateway_camera_id=s.get("camera_id"),
            enabled=True,
            description=f"从网关同步 camera_id={s.get('camera_id')}",
        )
        db.add(src)
        await db.flush()
        stream_manager.register(src.id, src.name, src.rtsp_url)
        asyncio.create_task(stream_manager.start(src.id))
        added += 1

    await db.commit()
    return {
        "added": added,
        "skipped": skipped,
        "total_from_gateway": len(streams),
        "gateway_url": gateway_url,
    }
