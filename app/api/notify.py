"""推送渠道配置 API：CRUD + 测试发送"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.notify_config import NotifyChannel
from app.services.notify_service import notify_service

router = APIRouter(prefix="/api/notify", tags=["notify"])

CHANNEL_TYPES = {"smtp", "wecom_bot", "wecom_app", "wxpusher", "qq_webhook"}


class ChannelCreate(BaseModel):
    name: str
    channel_type: str
    enabled: bool = True
    config: Dict[str, Any] = {}


class ChannelUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


@router.get("")
async def list_channels(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NotifyChannel).order_by(NotifyChannel.id))
    channels = result.scalars().all()
    return {"channels": [ch.to_dict() for ch in channels]}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_channel(body: ChannelCreate, db: AsyncSession = Depends(get_db)):
    if body.channel_type not in CHANNEL_TYPES:
        raise HTTPException(400, f"不支持的渠道类型，可选：{CHANNEL_TYPES}")

    ch = NotifyChannel(
        name=body.name,
        channel_type=body.channel_type,
        enabled=body.enabled,
    )
    ch.set_config(body.config)
    db.add(ch)
    await db.commit()
    await db.refresh(ch)
    return ch.to_dict()


@router.get("/{channel_id}")
async def get_channel(channel_id: int, db: AsyncSession = Depends(get_db)):
    ch = await db.get(NotifyChannel, channel_id)
    if not ch:
        raise HTTPException(404, "渠道不存在")
    return ch.to_dict()


@router.patch("/{channel_id}")
async def update_channel(
    channel_id: int, body: ChannelUpdate, db: AsyncSession = Depends(get_db)
):
    ch = await db.get(NotifyChannel, channel_id)
    if not ch:
        raise HTTPException(404, "渠道不存在")

    if body.name is not None:
        ch.name = body.name
    if body.enabled is not None:
        ch.enabled = body.enabled
    if body.config is not None:
        # 合并更新：将新 config 覆盖到旧 config，避免误删密码字段
        old_cfg = ch.get_config()
        for k, v in body.config.items():
            if v != "***":  # 跳过前端脱敏占位符
                old_cfg[k] = v
        ch.set_config(old_cfg)

    await db.commit()
    await db.refresh(ch)
    return ch.to_dict()


@router.delete("/{channel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_channel(channel_id: int, db: AsyncSession = Depends(get_db)):
    ch = await db.get(NotifyChannel, channel_id)
    if not ch:
        raise HTTPException(404, "渠道不存在")
    await db.delete(ch)
    await db.commit()


@router.post("/{channel_id}/test")
async def test_channel(channel_id: int, db: AsyncSession = Depends(get_db)):
    """发送测试消息到指定渠道"""
    ch = await db.get(NotifyChannel, channel_id)
    if not ch:
        raise HTTPException(404, "渠道不存在")

    ok, msg = await notify_service.send_test(channel_id)
    return {"ok": ok, "message": msg}
