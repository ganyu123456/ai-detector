"""报警记录模型"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, Integer, ForeignKey, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stream_id: Mapped[int] = mapped_column(Integer, ForeignKey("stream_sources.id", ondelete="CASCADE"), index=True)
    stream_name: Mapped[str] = mapped_column(String(100), default="")
    # type: yolo | intrusion | collision
    type: Mapped[str] = mapped_column(String(30), index=True)
    label: Mapped[str] = mapped_column(String(100), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    notified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stream_id": self.stream_id,
            "stream_name": self.stream_name,
            "type": self.type,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "snapshot_path": self.snapshot_path,
            "notified": self.notified,
            "created_at": (self.created_at.isoformat() + "Z") if self.created_at else None,
        }
