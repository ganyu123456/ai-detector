"""简单的 session cookie 认证，使用 HMAC-SHA256 签名，无需额外依赖。"""
import hashlib
import hmac
import time
from typing import Optional

from fastapi import Request
from fastapi.responses import RedirectResponse, JSONResponse

from app.config import settings

_SEP = "."
_MAX_AGE = 7 * 24 * 3600  # 7 天


def _sign(payload: str) -> str:
    return hmac.new(
        settings.SESSION_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def create_session_token(username: str) -> str:
    """生成带时间戳的签名 token：username.timestamp.signature"""
    ts = str(int(time.time()))
    payload = f"{username}{_SEP}{ts}"
    sig = _sign(payload)
    return f"{payload}{_SEP}{sig}"


def verify_session_token(token: str) -> Optional[str]:
    """验证 token，返回 username；无效或过期返回 None。"""
    try:
        parts = token.split(_SEP)
        if len(parts) != 3:
            return None
        username, ts, sig = parts
        payload = f"{username}{_SEP}{ts}"
        expected = _sign(payload)
        if not hmac.compare_digest(sig, expected):
            return None
        if int(time.time()) - int(ts) > _MAX_AGE:
            return None
        return username
    except Exception:
        return None


def get_session_user(request: Request) -> Optional[str]:
    """从 request cookie 中解析当前登录用户，未登录返回 None。"""
    token = request.cookies.get("session")
    if not token:
        return None
    return verify_session_token(token)


def check_credentials(username: str, password: str) -> bool:
    """校验用户名密码（常量时间比较，防时序攻击）。"""
    ok_user = hmac.compare_digest(username, settings.ADMIN_USERNAME)
    ok_pass = hmac.compare_digest(password, settings.ADMIN_PASSWORD)
    return ok_user and ok_pass
