"""
检测算法调度服务
订阅 RTSP 流帧（numpy BGR），按配置运行各检测器，触发报警并通过 WebSocket 广播检测框。
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebSocket 广播管理器（全局单例）
# ---------------------------------------------------------------------------

class WebSocketManager:
    """管理所有已连接的 WebSocket 客户端，广播 JSON 消息"""

    def __init__(self):
        self._clients: Set[asyncio.Queue] = set()

    def add_client(self) -> asyncio.Queue:
        """注册新客户端，返回其专属消息队列"""
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._clients.add(q)
        logger.debug(f"WS client connected, total={len(self._clients)}")
        return q

    def remove_client(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)
        logger.debug(f"WS client disconnected, total={len(self._clients)}")

    async def broadcast(self, message: dict) -> None:
        """向所有连接的客户端广播消息（丢帧不阻塞）"""
        if not self._clients:
            return
        payload = json.dumps(message, ensure_ascii=False)
        for q in list(self._clients):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # 客户端跟不上时静默丢弃，不影响检测流程


ws_manager = WebSocketManager()


# ---------------------------------------------------------------------------
# 检测工作者（单路流）
# ---------------------------------------------------------------------------

class DetectionWorker:
    """单路流的检测工作者，接收 numpy 帧，直接送入检测器"""

    def __init__(self, stream_id: int):
        self.stream_id = stream_id
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=settings.FRAME_QUEUE_SIZE)
        self._task: Optional[asyncio.Task] = None
        self._detectors: List = []
        self._detection_configs: List[dict] = []

    async def push_frame(self, stream_id: int, frame_np: np.ndarray) -> None:
        """接收 BGR numpy 帧，省去 JPEG 解码开销"""
        if not self._detectors:
            return
        try:
            self._queue.put_nowait(frame_np)
        except asyncio.QueueFull:
            pass

    async def start(self, detection_configs: List[dict]) -> None:
        self._detection_configs = detection_configs
        self._load_detectors()
        self._task = asyncio.create_task(
            self._run_loop(),
            name=f"detector-{self.stream_id}",
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for det in self._detectors:
            det.release()
        self._detectors.clear()

    def _load_detectors(self) -> None:
        from app.detectors.yolo_detector import YoloDetector
        from app.detectors.opencv_detector import IntrusionDetector, CollisionDetector

        self._detectors.clear()
        for cfg in self._detection_configs:
            if not cfg.get("enabled"):
                continue
            config = json.loads(cfg.get("config_json", "{}"))
            dtype = cfg.get("type", "")
            if dtype == "yolo":
                det = YoloDetector(config)
            elif dtype == "intrusion":
                det = IntrusionDetector(config)
            elif dtype == "collision":
                det = CollisionDetector(config)
            else:
                continue
            det._detection_type = dtype
            det._detection_config_id = cfg.get("id")
            det._detect_interval = float(config.get("detect_interval", 1.0))
            self._detectors.append(det)

    async def _run_loop(self) -> None:
        import cv2
        from app.services.alert_service import alert_service

        last_detect_ts: Dict[int, float] = {}

        while True:
            try:
                # 直接获取 numpy 帧，无需 JPEG 解码
                frame: np.ndarray = await self._queue.get()
                if frame is None or frame.size == 0:
                    continue

                for detector in self._detectors:
                    if not detector._initialized:
                        try:
                            detector.initialize()
                        except Exception as e:
                            logger.error(f"Detector init failed: {e}")
                            continue

                    det_id = id(detector)
                    now = time.time()
                    interval = getattr(detector, '_detect_interval', 1.0)
                    if now - last_detect_ts.get(det_id, 0) < interval:
                        continue
                    last_detect_ts[det_id] = now

                    try:
                        detections = detector.detect(frame)
                    except Exception as e:
                        logger.warning(f"Detection error: {e}")
                        continue

                    # 无论是否检测到目标，都向浏览器广播当前帧的检测结果
                    await ws_manager.broadcast({
                        "type": "detection",
                        "stream_id": self.stream_id,
                        "detections": [
                            {
                                "label": d.label,
                                "confidence": round(float(d.confidence), 3),
                                "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                            }
                            for d in detections
                        ],
                    })

                    if not detections:
                        continue

                    if alert_service.check_cooldown(self.stream_id, detector._detection_type):
                        continue

                    annotated = detector.draw(frame, detections)
                    _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    snapshot_bytes = jpeg_buf.tobytes()

                    top_det = max(detections, key=lambda d: d.confidence)
                    await alert_service.create(
                        stream_id=self.stream_id,
                        alert_type=detector._detection_type,
                        label=top_det.label,
                        confidence=top_det.confidence,
                        snapshot_bytes=snapshot_bytes,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Detection worker {self.stream_id} error: {e}")
                await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# 全局检测调度管理器
# ---------------------------------------------------------------------------

class DetectionManager:
    """全局检测调度管理器"""

    def __init__(self):
        self._workers: Dict[int, DetectionWorker] = {}

    async def start_stream(self, stream_id: int) -> None:
        from app.database import AsyncSessionLocal
        from app.models.detection import DetectionConfig
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DetectionConfig).where(
                    DetectionConfig.stream_id == stream_id,
                    DetectionConfig.enabled == True,
                )
            )
            configs = result.scalars().all()

        if not configs:
            return

        cfg_dicts = [
            {"id": c.id, "type": c.type, "enabled": c.enabled, "config_json": c.config_json}
            for c in configs
        ]

        await self.stop_stream(stream_id)
        worker = DetectionWorker(stream_id)
        self._workers[stream_id] = worker
        await worker.start(cfg_dicts)

        from app.services.stream_service import stream_manager
        state = stream_manager.get_state(stream_id)
        if state:
            state.add_frame_callback(worker.push_frame)

        logger.info(f"Detection started for stream {stream_id} ({len(cfg_dicts)} detectors)")

    async def stop_stream(self, stream_id: int) -> None:
        worker = self._workers.pop(stream_id, None)
        if worker:
            from app.services.stream_service import stream_manager
            state = stream_manager.get_state(stream_id)
            if state:
                state.remove_frame_callback(worker.push_frame)
            await worker.stop()

    async def reload_stream(self, stream_id: int) -> None:
        await self.stop_stream(stream_id)
        await self.start_stream(stream_id)

    async def start_all_enabled(self) -> None:
        from app.database import AsyncSessionLocal
        from app.models.detection import DetectionConfig
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DetectionConfig.stream_id).where(
                    DetectionConfig.enabled == True
                ).distinct()
            )
            stream_ids = [r[0] for r in result.all()]

        for sid in stream_ids:
            await self.start_stream(sid)


detection_manager = DetectionManager()
