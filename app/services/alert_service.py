"""报警服务：写入 SQLite、保存截图、触发通知，含冷却控制"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class AlertService:

    def __init__(self):
        # cooldown_key -> last_alert_ts
        self._last_alert: Dict[str, float] = {}

    def check_cooldown(self, stream_id: int, alert_type: str) -> bool:
        """返回 True 表示冷却未到期（应跳过报警），False 表示可以报警"""
        key = f"{stream_id}_{alert_type}"
        now = time.time()
        last = self._last_alert.get(key, 0)
        if now - last < settings.ALERT_COOLDOWN:
            return True
        self._last_alert[key] = now
        return False

    async def create(
        self,
        stream_id: int,
        alert_type: str,
        label: str,
        confidence: float,
        snapshot_bytes: Optional[bytes] = None,
    ) -> dict:
        from app.database import AsyncSessionLocal
        from app.models.alert import Alert
        from app.models.stream_source import StreamSource
        from sqlalchemy import select

        snapshot_path: Optional[str] = None
        if snapshot_bytes:
            snapshot_path = self._save_snapshot(stream_id, alert_type, snapshot_bytes)

        async with AsyncSessionLocal() as db:
            src_result = await db.execute(select(StreamSource).where(StreamSource.id == stream_id))
            src = src_result.scalar_one_or_none()
            stream_name = src.name if src else f"stream_{stream_id}"

            alert = Alert(
                stream_id=stream_id,
                stream_name=stream_name,
                type=alert_type,
                label=label,
                confidence=confidence,
                snapshot_path=snapshot_path,
                notified=False,
            )
            db.add(alert)
            await db.commit()
            await db.refresh(alert)
            alert_dict = alert.to_dict()

        logger.info(f"Alert: [{alert_type}] {label} ({confidence:.2f}) stream={stream_name}")

        import asyncio
        from app.services.notify_service import notify_service
        asyncio.create_task(notify_service.send_alert(alert_dict, snapshot_bytes))

        return alert_dict

    def _save_snapshot(self, stream_id: int, alert_type: str, data: bytes) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        day_dir: Path = settings.PICTURES_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%H%M%S_%f")[:12]
        filename = f"stream{stream_id}_{alert_type}_{ts}.jpg"
        filepath = day_dir / filename
        filepath.write_bytes(data)
        return str(filepath.relative_to(settings.DATA_DIR.parent))

    async def list_alerts(
        self,
        stream_id: Optional[int] = None,
        alert_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        from app.database import AsyncSessionLocal
        from app.models.alert import Alert
        from sqlalchemy import select, func, desc

        offset = (page - 1) * page_size

        async with AsyncSessionLocal() as db:
            q = select(Alert)
            count_q = select(func.count()).select_from(Alert)

            if stream_id:
                q = q.where(Alert.stream_id == stream_id)
                count_q = count_q.where(Alert.stream_id == stream_id)
            if alert_type:
                q = q.where(Alert.type == alert_type)
                count_q = count_q.where(Alert.type == alert_type)
            if start_time:
                from datetime import datetime as dt
                ts = dt.fromisoformat(start_time)
                q = q.where(Alert.created_at >= ts)
                count_q = count_q.where(Alert.created_at >= ts)
            if end_time:
                from datetime import datetime as dt
                ts = dt.fromisoformat(end_time)
                q = q.where(Alert.created_at <= ts)
                count_q = count_q.where(Alert.created_at <= ts)

            total = (await db.execute(count_q)).scalar()
            q = q.order_by(desc(Alert.created_at)).offset(offset).limit(page_size)
            items = (await db.execute(q)).scalars().all()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
            "items": [a.to_dict() for a in items],
        }

    async def delete_alert(self, alert_id: int) -> bool:
        from app.database import AsyncSessionLocal
        from app.models.alert import Alert

        async with AsyncSessionLocal() as db:
            alert = await db.get(Alert, alert_id)
            if not alert:
                return False
            if alert.snapshot_path:
                p = Path(alert.snapshot_path)
                if p.exists():
                    p.unlink(missing_ok=True)
            await db.delete(alert)
            await db.commit()
        return True

    async def get_stats(self) -> dict:
        from app.database import AsyncSessionLocal
        from app.models.alert import Alert
        from sqlalchemy import select, func

        async with AsyncSessionLocal() as db:
            total = (await db.execute(select(func.count()).select_from(Alert))).scalar()
            unnotified = (await db.execute(
                select(func.count()).select_from(Alert).where(Alert.notified == False)
            )).scalar()
            type_counts_result = await db.execute(
                select(Alert.type, func.count(Alert.id)).group_by(Alert.type)
            )
            type_counts = {row[0]: row[1] for row in type_counts_result.all()}

        return {"total": total, "unnotified": unnotified, "by_type": type_counts}


alert_service = AlertService()
