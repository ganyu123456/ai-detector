"""推送通知服务：HTTP Webhook + SMTP 邮件（可插拔扩展）"""
import asyncio
import logging
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import aiohttp

from app.config import settings

logger = logging.getLogger(__name__)


class NotifyService:

    async def send_alert(self, alert: dict, snapshot_bytes: Optional[bytes] = None) -> None:
        """并发发送所有已配置的通知渠道"""
        tasks = []
        if settings.WEBHOOK_URL:
            tasks.append(self._send_webhook(alert, snapshot_bytes))
        if settings.SMTP_ENABLED and settings.SMTP_USER and settings.SMTP_TO:
            tasks.append(self._send_email(alert, snapshot_bytes))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ── Webhook ────────────────────────────────────────────────────────────
    async def _send_webhook(self, alert: dict, snapshot_bytes: Optional[bytes]) -> None:
        """发送 HTTP POST Webhook（企业微信机器人/飞书/钉钉 均使用此格式）"""
        payload = {
            "msgtype": "text",
            "text": {
                "content": (
                    f"⚠️ AI检测报警\n"
                    f"流：{alert.get('stream_name', '未知')}\n"
                    f"类型：{alert.get('type', '')}\n"
                    f"目标：{alert.get('label', '')}\n"
                    f"置信度：{alert.get('confidence', 0):.1%}\n"
                    f"时间：{alert.get('created_at', '')}"
                )
            },
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.WEBHOOK_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Webhook sent for alert {alert.get('id')}")
                    else:
                        logger.warning(f"Webhook returned {resp.status}")
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")

    # ── SMTP 邮件 ──────────────────────────────────────────────────────────
    async def _send_email(self, alert: dict, snapshot_bytes: Optional[bytes]) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._send_email_sync, alert, snapshot_bytes)
        except Exception as e:
            logger.error(f"Email send failed: {e}")

    def _send_email_sync(self, alert: dict, snapshot_bytes: Optional[bytes]) -> None:
        alert_type = alert.get("type", "未知")
        stream_name = alert.get("stream_name", "未知")
        label = alert.get("label", "")
        confidence = alert.get("confidence", 0)
        created_at = alert.get("created_at", "")

        subject = f"[AI检测报警] {stream_name} - {alert_type} 检测到 {label}"
        body = f"""
        <html><body>
        <h2 style="color:#e74c3c;">⚠️ AI 视频检测报警通知</h2>
        <table border="1" cellpadding="8" style="border-collapse:collapse;">
          <tr><td><b>视频流</b></td><td>{stream_name}</td></tr>
          <tr><td><b>报警类型</b></td><td>{alert_type}</td></tr>
          <tr><td><b>检测目标</b></td><td>{label}</td></tr>
          <tr><td><b>置信度</b></td><td>{confidence:.1%}</td></tr>
          <tr><td><b>报警时间</b></td><td>{created_at}</td></tr>
        </table>
        <p>请及时查看 AI 检测平台。</p>
        </body></html>
        """

        msg = MIMEMultipart("mixed")
        msg["From"] = settings.SMTP_USER
        msg["To"] = settings.SMTP_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html", "utf-8"))

        if snapshot_bytes:
            part = MIMEBase("image", "jpeg")
            part.set_payload(snapshot_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename="snapshot.jpg")
            msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, context=context) as server:
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_USER, settings.SMTP_TO.split(","), msg.as_string())

        logger.info(f"Alert email sent to {settings.SMTP_TO}")


notify_service = NotifyService()
