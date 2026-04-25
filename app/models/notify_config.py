"""通知渠道配置模型"""
import json
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class NotifyChannel(Base):
    """持久化存储各类推送渠道配置"""
    __tablename__ = "notify_channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(100))
    """渠道显示名称，如"企业微信报警机器人"."""

    channel_type: Mapped[str] = mapped_column(String(30))
    """渠道类型：smtp | wecom_bot | wecom_app | wxpusher | qq_webhook"""

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    config_json: Mapped[str] = mapped_column(Text, default="{}")
    """渠道私有配置（JSON 字符串），各类型字段说明：

    smtp:
      host, port, user, password, to (逗号分隔), use_ssl(bool)

    wecom_bot（企业微信机器人）:
      webhook_url

    wecom_app（企业微信应用）:
      corpid, corpsecret, agentid, touser("@all" 或 uid 逗号分隔)

    wxpusher（微信个人推送）:
      app_token, uids (逗号分隔的 UID 列表)

    qq_webhook（QQ 机器人/自定义 webhook）:
      webhook_url, msg_template(可选)
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def get_config(self) -> Dict[str, Any]:
        try:
            return json.loads(self.config_json or "{}")
        except Exception:
            return {}

    def set_config(self, cfg: Dict[str, Any]) -> None:
        self.config_json = json.dumps(cfg, ensure_ascii=False)

    def to_dict(self) -> dict:
        cfg = self.get_config()
        # 脱敏：隐藏密码和 token 明文
        safe_cfg = {}
        for k, v in cfg.items():
            if any(kw in k.lower() for kw in ("password", "secret", "token")):
                safe_cfg[k] = "***" if v else ""
            else:
                safe_cfg[k] = v
        return {
            "id": self.id,
            "name": self.name,
            "channel_type": self.channel_type,
            "enabled": self.enabled,
            "config": safe_cfg,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
