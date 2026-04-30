"""
WebSocket 事件广播端点
浏览器连接 /ws/events，接收 JSON 检测结果实时推送。
消息格式：
  {"type":"detection","stream_id":1,"detections":[{"label":"person","confidence":0.91,"bbox":[x1,y1,x2,y2]}]}
"""
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.detection_service import ws_manager

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """实时检测结果广播（所有流共享一个连接）"""
    await websocket.accept()
    queue = ws_manager.add_client()
    logger.info("WebSocket client connected to /ws/events")
    try:
        while True:
            # 等待广播消息，同时保持心跳检测
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(payload)
            except asyncio.TimeoutError:
                # 发送 ping 心跳，保持连接活跃
                await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/events")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        ws_manager.remove_client(queue)
