"""推送通知服务：从数据库读取渠道配置，支持多渠道并发发送。

支持渠道：
  smtp       - SMTP 邮件（QQ邮箱/163/Gmail 等）
  wecom_bot  - 企业微信机器人（Webhook）
  wecom_app  - 企业微信应用消息
  wxpusher   - 微信个人推送（WxPusher）
  qq_webhook - QQ 机器人 / 自定义 Webhook
"""
import asyncio
import logging
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class NotifyService:

    async def send_alert(
        self,
        alert: dict,
        snapshot_bytes: Optional[bytes] = None,
    ) -> None:
        """读取数据库中所有启用的渠道，并发发送报警通知。"""
        channels = await self._load_enabled_channels()
        if not channels:
            return

        tasks = [
            self._dispatch(ch, alert, snapshot_bytes)
            for ch in channels
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ch, res in zip(channels, results):
            if isinstance(res, Exception):
                logger.error(f"Channel [{ch['name']}] send failed: {res}")

    async def send_test(self, channel_id: int) -> tuple[bool, str]:
        """向指定渠道发送测试消息，返回 (ok, message)。"""
        channels = await self._load_enabled_channels(channel_id=channel_id)
        if not channels:
            return False, "渠道不存在或已禁用"
        ch = channels[0]
        test_alert = {
            "id": 0,
            "stream_name": "测试流",
            "type": "test",
            "label": "测试",
            "confidence": 0.99,
            "created_at": "2026-01-01 00:00:00",
        }
        try:
            await self._dispatch(ch, test_alert, None)
            return True, "发送成功"
        except Exception as e:
            return False, str(e)

    # ── 内部分发 ────────────────────────────────────────────────────

    async def _dispatch(
        self,
        ch: dict,
        alert: dict,
        snapshot_bytes: Optional[bytes],
    ) -> None:
        t = ch["channel_type"]
        cfg = ch["config"]
        if t == "smtp":
            await self._send_smtp(cfg, alert, snapshot_bytes)
        elif t == "wecom_bot":
            await self._send_wecom_bot(cfg, alert)
        elif t == "wecom_app":
            await self._send_wecom_app(cfg, alert)
        elif t == "wxpusher":
            await self._send_wxpusher(cfg, alert)
        elif t == "qq_webhook":
            await self._send_qq_webhook(cfg, alert)
        else:
            logger.warning(f"Unknown channel type: {t}")

    # ── SMTP 邮件 ────────────────────────────────────────────────────

    async def _send_smtp(
        self, cfg: dict, alert: dict, snapshot_bytes: Optional[bytes]
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_smtp_sync, cfg, alert, snapshot_bytes)

    def _send_smtp_sync(
        self, cfg: dict, alert: dict, snapshot_bytes: Optional[bytes]
    ) -> None:
        host = cfg.get("host", "smtp.qq.com")
        port = int(cfg.get("port", 465))
        user = cfg.get("user", "")
        password = cfg.get("password", "")
        to = cfg.get("to", "")
        use_ssl = cfg.get("use_ssl", True)

        if not user or not password or not to:
            raise ValueError("SMTP 配置不完整（需要 user / password / to）")

        subject = (
            f"[AI检测报警] {alert.get('stream_name','未知')} - "
            f"{alert.get('type','')} 检测到 {alert.get('label','')}"
        )
        body = f"""<html><body>
<h2 style="color:#e74c3c;">⚠️ AI 视频检测报警通知</h2>
<table border="1" cellpadding="8" style="border-collapse:collapse;">
  <tr><td><b>视频流</b></td><td>{alert.get('stream_name','未知')}</td></tr>
  <tr><td><b>报警类型</b></td><td>{alert.get('type','')}</td></tr>
  <tr><td><b>检测目标</b></td><td>{alert.get('label','')}</td></tr>
  <tr><td><b>置信度</b></td><td>{alert.get('confidence',0):.1%}</td></tr>
  <tr><td><b>报警时间</b></td><td>{alert.get('created_at','')}</td></tr>
</table>
<p>请及时查看 AI 检测平台。</p>
</body></html>"""

        msg = MIMEMultipart("mixed")
        msg["From"] = user
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html", "utf-8"))

        if snapshot_bytes:
            part = MIMEBase("image", "jpeg")
            part.set_payload(snapshot_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename="snapshot.jpg")
            msg.attach(part)

        ctx = ssl.create_default_context()
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, context=ctx) as server:
                server.login(user, password)
                server.sendmail(user, to.split(","), msg.as_string())
        else:
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.login(user, password)
                server.sendmail(user, to.split(","), msg.as_string())

        logger.info(f"SMTP alert sent to {to}")

    # ── 企业微信机器人 ───────────────────────────────────────────────

    async def _send_wecom_bot(self, cfg: dict, alert: dict) -> None:
        webhook_url = cfg.get("webhook_url", "")
        if not webhook_url:
            raise ValueError("企业微信机器人 webhook_url 未配置")

        content = (
            f"⚠️ **AI检测报警**\n"
            f"> 视频流：{alert.get('stream_name','未知')}\n"
            f"> 报警类型：{alert.get('type','')}\n"
            f"> 检测目标：{alert.get('label','')}\n"
            f"> 置信度：{alert.get('confidence',0):.1%}\n"
            f"> 时间：{alert.get('created_at','')}"
        )
        payload = {"msgtype": "markdown", "markdown": {"content": content}}

        async with aiohttp.ClientSession() as s:
            async with s.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.json()
                if body.get("errcode", 0) != 0:
                    raise RuntimeError(f"WeCom bot error: {body}")

        logger.info("WeChat Work bot alert sent")

    # ── 企业微信应用消息 ─────────────────────────────────────────────

    async def _send_wecom_app(self, cfg: dict, alert: dict) -> None:
        corpid = cfg.get("corpid", "")
        corpsecret = cfg.get("corpsecret", "")
        agentid = cfg.get("agentid", "")
        touser = cfg.get("touser", "@all")

        if not corpid or not corpsecret or not agentid:
            raise ValueError("企业微信应用配置不完整（需要 corpid / corpsecret / agentid）")

        async with aiohttp.ClientSession() as s:
            # 获取 access_token
            token_url = (
                f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
                f"?corpid={corpid}&corpsecret={corpsecret}"
            )
            async with s.get(token_url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                t = await r.json()
                if t.get("errcode", 0) != 0:
                    raise RuntimeError(f"WeCom gettoken error: {t}")
                access_token = t["access_token"]

            # 发送消息
            send_url = (
                "https://qyapi.weixin.qq.com/cgi-bin/message/send"
                f"?access_token={access_token}"
            )
            content = (
                f"⚠️ AI检测报警\n"
                f"视频流：{alert.get('stream_name','未知')}\n"
                f"报警类型：{alert.get('type','')}\n"
                f"检测目标：{alert.get('label','')}\n"
                f"置信度：{alert.get('confidence',0):.1%}\n"
                f"时间：{alert.get('created_at','')}"
            )
            payload = {
                "touser": touser,
                "msgtype": "text",
                "agentid": int(agentid),
                "text": {"content": content},
                "safe": 0,
            }
            async with s.post(
                send_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                body = await r.json()
                if body.get("errcode", 0) != 0:
                    raise RuntimeError(f"WeCom app send error: {body}")

        logger.info(f"WeChat Work app message sent to {touser}")

    # ── 微信个人推送（WxPusher）──────────────────────────────────────

    async def _send_wxpusher(self, cfg: dict, alert: dict) -> None:
        app_token = cfg.get("app_token", "")
        uids = cfg.get("uids", "")  # 逗号分隔的 UID

        if not app_token or not uids:
            raise ValueError("WxPusher 配置不完整（需要 app_token / uids）")

        uid_list = [u.strip() for u in uids.split(",") if u.strip()]
        content = (
            f"<h3>⚠️ AI检测报警</h3>"
            f"<p><b>视频流</b>：{alert.get('stream_name','未知')}</p>"
            f"<p><b>类型</b>：{alert.get('type','')}</p>"
            f"<p><b>目标</b>：{alert.get('label','')}</p>"
            f"<p><b>置信度</b>：{alert.get('confidence',0):.1%}</p>"
            f"<p><b>时间</b>：{alert.get('created_at','')}</p>"
        )
        payload = {
            "appToken": app_token,
            "content": content,
            "contentType": 2,  # 2 = HTML
            "uids": uid_list,
        }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                "https://wxpusher.zjiecode.com/api/send/message",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.json()
                if not body.get("success", False):
                    raise RuntimeError(f"WxPusher error: {body.get('msg')}")

        logger.info(f"WxPusher alert sent to {uid_list}")

    # ── QQ / 自定义 Webhook ──────────────────────────────────────────

    async def _send_qq_webhook(self, cfg: dict, alert: dict) -> None:
        webhook_url = cfg.get("webhook_url", "")
        if not webhook_url:
            raise ValueError("QQ Webhook URL 未配置")

        msg_template = cfg.get(
            "msg_template",
            "⚠️ AI检测报警\n流：{stream_name}\n类型：{type}\n目标：{label}\n置信度：{confidence:.1%}\n时间：{created_at}",
        )
        try:
            content = msg_template.format(
                stream_name=alert.get("stream_name", "未知"),
                type=alert.get("type", ""),
                label=alert.get("label", ""),
                confidence=alert.get("confidence", 0),
                created_at=alert.get("created_at", ""),
            )
        except Exception:
            content = f"AI检测报警 - {alert.get('stream_name','未知')}"

        payload = {"message": content, "content": content, "text": content}

        async with aiohttp.ClientSession() as s:
            async with s.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"QQ Webhook returned {resp.status}")

        logger.info("QQ Webhook alert sent")

    # ── 工具：从 DB 读取渠道 ─────────────────────────────────────────

    async def _load_enabled_channels(
        self, channel_id: Optional[int] = None
    ) -> List[dict]:
        from sqlalchemy import select
        from app.database import AsyncSessionLocal
        from app.models.notify_config import NotifyChannel

        async with AsyncSessionLocal() as db:
            stmt = select(NotifyChannel).where(NotifyChannel.enabled == True)
            if channel_id is not None:
                stmt = select(NotifyChannel).where(NotifyChannel.id == channel_id)
            result = await db.execute(stmt)
            channels = result.scalars().all()
            return [
                {
                    "name": ch.name,
                    "channel_type": ch.channel_type,
                    "config": ch.get_config(),
                }
                for ch in channels
            ]


notify_service = NotifyService()
