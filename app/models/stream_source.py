"""流源模型：管理 RTSP 流地址（与具体摄像头品牌解耦）"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class StreamSource(Base):
    __tablename__ = "stream_sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))
    # RTSP 地址，如 rtsp://192.168.1.1:8554/camera1
    rtsp_url: Mapped[str] = mapped_column(String(500))
    # 可选：关联到 miloco-camera 网关的摄像头 ID
    gateway_camera_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # 备注
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rtsp_url": self.rtsp_url,
            "gateway_camera_id": self.gateway_camera_id,
            "enabled": self.enabled,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
