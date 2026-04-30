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

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    stream_id: int
    name: str
    rtsp_url: str
    status: str = "stopped"       # stopped | starting | running | error
    error_msg: str = ""
    latest_frame: Optional[bytes] = None          # JPEG bytes，仅供 snapshot 端点使用
    latest_np: Optional[np.ndarray] = None        # 最新 BGR numpy 帧
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
        if state.status == "running":
            return True

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
        state.latest_frame = None
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
                try:
                    container = av.open(state.rtsp_url, options=rtsp_opts)
                    vstream = container.streams.video[0]

                    hw_ctx = None
                    if use_hw and not force_sw:
                        hw_ctx = _try_create_hw_context(decoder_name, vstream)

                    _decode_loop(container, vstream, hw_ctx, cv2)
                    container.close()
                    if hw_ctx is not None:
                        hw_ctx.close()
                    break
                except Exception as e:
                    if attempt == 0 and use_hw:
                        logger.warning(
                            f"Stream {state.stream_id} hw decode error: {e}, retrying SW"
                        )
                        continue
                    asyncio.run_coroutine_threadsafe(_set_error(str(e)), loop)
                    break

        def _decode_loop(container, vstream, hw_ctx, cv2):
            """解码循环：优先使用独立 hw_ctx 解码，否则用容器默认解码"""
            for packet in container.demux(vstream):
                if state.status not in ("running", "starting"):
                    break
                try:
                    decoder = hw_ctx if hw_ctx is not None else vstream.codec_context
                    for frame in decoder.decode(packet):
                        bgr: np.ndarray = frame.to_ndarray(format="bgr24")

                        # 编码 JPEG 供 snapshot 端点使用（每帧一次，成本低于 MJPEG 推流）
                        ret, jpeg = cv2.imencode(
                            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                        jpeg_bytes = jpeg.tobytes() if ret else None

                        asyncio.run_coroutine_threadsafe(
                            _dispatch_frame(bgr, jpeg_bytes), loop
                        )
                except Exception:
                    continue

        async def _dispatch_frame(bgr: np.ndarray, jpeg_bytes: Optional[bytes]) -> None:
            state.latest_np = bgr
            if jpeg_bytes:
                state.latest_frame = jpeg_bytes
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

        async def _set_error(msg: str) -> None:
            state.status = "error"
            state.error_msg = msg
            logger.error(f"Stream {state.stream_id} error: {msg}")

        try:
            state.status = "running"
            logger.info(f"Stream {state.stream_id} ({state.rtsp_url}) started")
            await loop.run_in_executor(None, _pull_and_decode)
        except asyncio.CancelledError:
            logger.info(f"Stream {state.stream_id} cancelled")
        except Exception as e:
            state.status = "error"
            state.error_msg = str(e)
            logger.error(f"Stream {state.stream_id} error: {e}")
        finally:
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


stream_manager = StreamManager()
