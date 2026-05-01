"""
RTSP 流服务：从 MediaMTX 拉取 RTSP 视频流，通过独立 CodecContext 实现 NVDEC 硬解码，
将 numpy 帧直接分发给检测服务（不再做 JPEG 编码），降低 CPU 占用。
latest_frame（JPEG）仅供 snapshot 端点使用，按需编码。
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    stream_id: int
    name: str
    rtsp_url: str
    status: str = "stopped"       # stopped | starting | running | error
    error_msg: str = ""
    latest_np: Optional[np.ndarray] = None        # 最新 BGR numpy 帧（snapshot 按需编码 JPEG）
    frame_event: asyncio.Event = field(default_factory=asyncio.Event)
    fps: float = 0.0
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _frame_callbacks: List[Callable] = field(default_factory=list, repr=False)
    _frame_count: int = field(default=0, repr=False)
    _fps_ts: float = field(default_factory=time.monotonic, repr=False)

    def add_frame_callback(self, cb: Callable) -> None:
        if cb not in self._frame_callbacks:
            self._frame_callbacks.append(cb)

    def remove_frame_callback(self, cb: Callable) -> None:
        if cb in self._frame_callbacks:
            self._frame_callbacks.remove(cb)

    def update_fps(self) -> None:
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_ts
        if elapsed >= 5.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._fps_ts = now


class StreamManager:
    """管理所有 RTSP 流的拉取和帧分发"""

    def __init__(self):
        self._streams: Dict[int, StreamState] = {}

    def register(self, stream_id: int, name: str, rtsp_url: str) -> StreamState:
        if stream_id not in self._streams:
            self._streams[stream_id] = StreamState(
                stream_id=stream_id, name=name, rtsp_url=rtsp_url
            )
        return self._streams[stream_id]

    def unregister(self, stream_id: int) -> None:
        self._streams.pop(stream_id, None)

    def get_state(self, stream_id: int) -> Optional[StreamState]:
        return self._streams.get(stream_id)

    def all_states(self) -> List[StreamState]:
        return list(self._streams.values())

    async def start(self, stream_id: int) -> bool:
        state = self._streams.get(stream_id)
        if not state:
            return False

        # 正常运行中直接返回，不重复启动
        if state.status == "running" and state._task and not state._task.done():
            return True

        # 若有旧 Task 仍在运行（如 error 重试等待中），先取消，避免并发重连
        if state._task and not state._task.done():
            state._task.cancel()
            try:
                await state._task
            except (asyncio.CancelledError, Exception):
                pass

        state.status = "starting"
        state.error_msg = ""
        state._task = asyncio.create_task(
            self._run_stream(state), name=f"stream-{stream_id}"
        )
        return True

    async def stop(self, stream_id: int) -> None:
        state = self._streams.get(stream_id)
        if not state:
            return
        if state._task and not state._task.done():
            state._task.cancel()
            try:
                await state._task
            except (asyncio.CancelledError, Exception):
                pass
        state.status = "stopped"
        state.latest_np = None
        logger.info(f"Stream {stream_id} stopped")

    async def start_all_enabled(self) -> None:
        from app.database import AsyncSessionLocal
        from app.models.stream_source import StreamSource
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(StreamSource).where(StreamSource.enabled == True))
            sources = result.scalars().all()

        for src in sources:
            self.register(src.id, src.name, src.rtsp_url)
            await self.start(src.id)
            logger.info(f"Auto-started stream: {src.name} ({src.rtsp_url})")

    async def _run_stream(self, state: StreamState) -> None:
        loop = asyncio.get_running_loop()

        def _pull_and_decode():
            import av
            import cv2

            decoder_name, use_hw = _pick_video_decoder()
            logger.info(
                f"Stream {state.stream_id}: opening {state.rtsp_url} "
                f"decoder={decoder_name} hw={use_hw}"
            )

            rtsp_opts = {
                "rtsp_transport": "tcp",
                "stimeout": "5000000",
                "fflags": "nobuffer",
                "flags": "low_delay",
            }

            # 尝试硬解优先，失败自动降级软解
            for attempt, force_sw in enumerate([False, True]):
                if attempt == 1:
                    logger.info(f"Stream {state.stream_id}: falling back to software decode")
                hw_ctx = None
                container = None
                try:
                    container = av.open(state.rtsp_url, options=rtsp_opts)
                    vstream = container.streams.video[0]

                    if use_hw and not force_sw:
                        hw_ctx = _try_create_hw_context(decoder_name, vstream)

                    _decode_loop(container, vstream, hw_ctx, cv2)
                    break
                except Exception as e:
                    if attempt == 0 and use_hw:
                        logger.warning(
                            f"Stream {state.stream_id} hw decode error: {e}, retrying SW"
                        )
                        continue
                    # 通知外层重连循环，抛出异常而非调用 _set_error
                    raise
                finally:
                    # 安全关闭资源（CodecContext 不一定有 close 方法）
                    if hw_ctx is not None:
                        _safe_close(hw_ctx)
                    if container is not None:
                        try:
                            container.close()
                        except Exception:
                            pass

        def _decode_loop(container, vstream, hw_ctx, cv2):
            """解码循环：优先使用独立 hw_ctx 解码，否则用容器默认解码。
            帧采样节流：仅当距上次 to_ndarray 超过 FRAME_SAMPLE_INTERVAL 才执行
            GPU→CPU 内存拷贝，其余帧由 NVDEC 解码后直接丢弃，大幅减少 PCIe DMA 次数。
            """
            from app.config import settings as _cfg
            sample_interval: float = _cfg.FRAME_SAMPLE_INTERVAL
            last_sample_ts: float = 0.0

            for packet in container.demux(vstream):
                if state.status not in ("running", "starting"):
                    break
                try:
                    decoder = hw_ctx if hw_ctx is not None else vstream.codec_context
                    for frame in decoder.decode(packet):
                        now = time.monotonic()
                        if now - last_sample_ts < sample_interval:
                            continue  # 丢弃此帧，跳过 GPU→CPU 拷贝
                        last_sample_ts = now
                        bgr: np.ndarray = frame.to_ndarray(format="bgr24")
                        asyncio.run_coroutine_threadsafe(
                            _dispatch_frame(bgr), loop
                        )
                except Exception:
                    continue

        async def _dispatch_frame(bgr: np.ndarray) -> None:
            state.latest_np = bgr
            state.update_fps()
            # 通知 snapshot 等待者（原 MJPEG generator 已移除）
            state.frame_event.set()
            state.frame_event.clear()
            # 将 numpy 帧分发给检测 worker
            for cb in list(state._frame_callbacks):
                try:
                    await cb(state.stream_id, bgr)
                except Exception as e:
                    logger.warning(f"Stream {state.stream_id} frame callback error: {e}")

        # ── 自动重连循环 ────────────────────────────────────────────────────
        from app.services.notify_service import notify_service

        retry_delay      = settings.STREAM_RETRY_DELAY
        notify_max       = settings.OFFLINE_NOTIFY_MAX
        notify_interval  = settings.OFFLINE_NOTIFY_INTERVAL
        attempt_count    = 0
        notify_count     = 0          # 已发离线通知次数
        last_notify_ts   = 0.0        # 上次离线通知的时间戳

        while True:
            try:
                state.status = "running"
                state.error_msg = ""
                if attempt_count == 0:
                    logger.info(f"Stream {state.stream_id} ({state.rtsp_url}) started")
                else:
                    logger.info(
                        f"Stream {state.stream_id} reconnecting (attempt #{attempt_count})"
                    )
                await loop.run_in_executor(None, _pull_and_decode)

                # _pull_and_decode 正常返回（收到 EOF / RTSP Teardown）
                # 不代表用户主动停止，应继续重连，除非已被手动 stop
                if state.status == "stopped":
                    logger.info(f"Stream {state.stream_id} was stopped, exiting")
                    break

                # 流曾成功运行后意外断开：若之前发过离线通知，先发恢复通知
                if notify_count > 0:
                    try:
                        await notify_service.send_system_event(
                            event_type="stream_online",
                            title=f"📷 摄像头已恢复：{state.name}",
                            detail=(
                                f"流地址：{state.rtsp_url}\n"
                                f"共重连 {attempt_count} 次后恢复正常"
                            ),
                        )
                        logger.info(f"Stream {state.stream_id} recovery notification sent")
                        notify_count = 0       # 重置，下次离线重新计数
                        last_notify_ts = 0.0
                    except Exception as ne:
                        logger.warning(f"Stream {state.stream_id} recovery notify failed: {ne}")

                # 视为意外断开，触发重连（与抛异常时行为一致）
                attempt_count += 1
                state.status = "error"
                logger.warning(
                    f"Stream {state.stream_id} ended unexpectedly (EOF/disconnect), "
                    f"retrying in {retry_delay}s (attempt #{attempt_count})"
                )
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    logger.info(f"Stream {state.stream_id} cancelled during retry wait")
                    break
                if state.status == "stopped":
                    break
                continue

            except asyncio.CancelledError:
                logger.info(f"Stream {state.stream_id} cancelled")
                break

            except Exception as e:
                attempt_count += 1
                state.status = "error"
                state.error_msg = str(e)
                logger.error(
                    f"Stream {state.stream_id} error: {e} — "
                    f"retrying in {retry_delay}s (attempt #{attempt_count})"
                )

                # 离线通知：最多 notify_max 次，每次间隔 notify_interval 秒
                now_ts = time.monotonic()
                if (
                    notify_max > 0
                    and notify_count < notify_max
                    and (now_ts - last_notify_ts) >= notify_interval
                ):
                    try:
                        remaining = notify_max - notify_count - 1
                        await notify_service.send_system_event(
                            event_type="stream_offline",
                            title=f"📵 摄像头离线：{state.name}",
                            detail=(
                                f"流地址：{state.rtsp_url}\n"
                                f"错误原因：{e}\n"
                                f"第 {notify_count + 1} 次通知"
                                + (f"，还将最多再通知 {remaining} 次" if remaining > 0 else "，本次为最后一次通知")
                            ),
                        )
                        notify_count += 1
                        last_notify_ts = now_ts
                        logger.info(
                            f"Stream {state.stream_id} offline notification sent "
                            f"({notify_count}/{notify_max})"
                        )
                    except Exception as ne:
                        logger.warning(f"Stream {state.stream_id} offline notify failed: {ne}")

                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    logger.info(f"Stream {state.stream_id} cancelled during retry wait")
                    break
                # 检查是否在等待期间被手动停止
                if state.status == "stopped":
                    break

        if state.status not in ("stopped",):
            state.status = "error"


def _try_create_hw_context(decoder_name: str, vstream):
    """
    创建独立的 NVDEC CodecContext。
    复制 extradata（SPS/PPS）确保解码器正确初始化。
    返回 None 表示创建失败，调用方应降级到软解。
    """
    try:
        import av
        hw_ctx = av.codec.CodecContext.create(decoder_name, "r")
        # 将容器流的 extradata（SPS/PPS）复制给独立解码器
        src_extra = vstream.codec_context.extradata
        if src_extra:
            hw_ctx.extradata = src_extra
        hw_ctx.open()
        logger.info(f"NVDEC hardware decoder '{decoder_name}' initialized successfully")
        return hw_ctx
    except Exception as e:
        logger.warning(f"Failed to create hw CodecContext '{decoder_name}': {e}")
        return None


def _pick_video_decoder() -> tuple[str, bool]:
    """
    探测最优视频解码器。
    返回 (decoder_name, use_hardware)。
    使用 PyAV 内置 codec 注册表，不依赖外部 ffmpeg 命令。
    """
    try:
        import av
        available: set = av.codec.codecs_available

        # NVIDIA CUVID（Windows/Linux）
        if "hevc_cuvid" in available:
            return "hevc_cuvid", True
        if "h264_cuvid" in available:
            return "h264_cuvid", True

        # VA-API（Linux）
        if "hevc_vaapi" in available:
            return "hevc_vaapi", True

        # VideoToolbox（macOS）
        if "hevc_videotoolbox" in available:
            return "hevc_videotoolbox", True

        # V4L2（嵌入式 Linux）
        if "hevc_v4l2m2m" in available:
            return "hevc_v4l2m2m", True

    except Exception as e:
        logger.debug(f"Codec probe failed: {e}")

    return "hevc", False


def _safe_close(obj) -> None:
    """安全关闭 PyAV 对象：部分版本的 CodecContext 没有 close() 方法，兼容处理。"""
    close_fn = getattr(obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as e:
            logger.debug(f"_safe_close ignored: {e}")


stream_manager = StreamManager()
