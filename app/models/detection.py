"""检测配置模型：每路流可独立配置检测算法"""
from datetime import datetime
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DetectionConfig(Base):
    __tablename__ = "detection_configs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stream_id: Mapped[int] = mapped_column(Integer, ForeignKey("stream_sources.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    # type: yolo | intrusion | collision
    type: Mapped[str] = mapped_column(String(30))
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # JSON 配置，不同类型不同字段：
    #   yolo:      {"model": "yolo11n.pt", "confidence": 0.5, "classes": ["person"], "iou": 0.45, "detect_interval": 1.0}
    #   intrusion: {"roi": [[x,y],...], "min_area": 500, "sensitivity": 50, "detect_interval": 1.0}
    #   collision: {"lines": [[[x1,y1],[x2,y2]],...], "direction": "any", "detect_interval": 1.0}
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "stream_id": self.stream_id,
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "config": json.loads(self.config_json or "{}"),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
