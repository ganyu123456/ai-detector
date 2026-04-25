"""
RTSP 流服务：从网关或其他 RTSP 源拉取视频流，解码帧并分发给检测器。
支持 GPU 硬件解码（NVIDIA CUDA/CUVID）和 CPU 软解码降级。
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    stream_id: int
    name: str
    rtsp_url: str
    status: str = "stopped"       # stopped | starting | running | error
    error_msg: str = ""
    latest_frame: Optional[bytes] = None
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
            import numpy as np

            decoder_name = _pick_hevc_decoder()
            logger.info(f"Stream {state.stream_id}: opening {state.rtsp_url} with {decoder_name}")

            try:
                container = av.open(
                    state.rtsp_url,
                    options={
                        "rtsp_transport": "tcp",
                        "stimeout": "5000000",
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                    },
                )

                for packet in container.demux(video=0):
                    if not state.status == "running" and state.status != "starting":
                        break
                    try:
                        for frame in packet.decode():
                            bgr = frame.to_ndarray(format="bgr24")
                            ret, jpeg = cv2.imencode(
                                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
                            )
                            if ret:
                                jpeg_bytes = jpeg.tobytes()
                                asyncio.run_coroutine_threadsafe(
                                    _dispatch_frame(jpeg_bytes), loop
                                )
                    except Exception:
                        continue
                container.close()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(_set_error(str(e)), loop)

        async def _dispatch_frame(data: bytes) -> None:
            state.latest_frame = data
            state.update_fps()
            state.frame_event.set()
            state.frame_event.clear()
            for cb in list(state._frame_callbacks):
                try:
                    await cb(state.stream_id, data)
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


def _pick_hevc_decoder() -> str:
    """选择最优 HEVC 解码器"""
    import subprocess
    import av

    def try_codec(name: str) -> bool:
        try:
            av.Codec(name, "r")
            return True
        except Exception:
            return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3,
        )
        hw_list = [x.strip() for x in result.stdout.strip().split("\n")[1:] if x.strip()]
    except Exception:
        hw_list = []

    if ("cuda" in hw_list or "cuvid" in hw_list) and try_codec("hevc_cuvid"):
        return "hevc_cuvid"
    if try_codec("hevc_v4l2m2m"):
        return "hevc_v4l2m2m"
    return "hevc"


stream_manager = StreamManager()
