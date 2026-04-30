"""YOLO 目标检测器（基于 ultralytics YOLOv8/v11）

推理后端优先级（自动选择）：
  1. ONNX + onnxruntime-gpu  — EXE 默认路径，体积最小，支持 CUDA
  2. ultralytics YOLO        — 源码运行 / Docker 路径
device 优先级（auto 模式）：
  CUDA → CPU
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.detectors.base import AbstractDetector, Detection
from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """将 'auto' 解析为实际可用设备，其余原样返回。"""
    if requested != "auto":
        return requested
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _onnx_path_for(model_path: str) -> Optional[Path]:
    """返回对应的 .onnx 路径（若存在）。

    查找顺序：
    1. model_path 同级目录（如 data/models/yolo11n.onnx）
    2. settings.MODELS_DIR
    3. PyInstaller 打包目录 sys._MEIPASS/data/models/（EXE 运行时）
    """
    import sys
    p = Path(model_path)
    candidate = p.with_suffix(".onnx")
    if candidate.exists():
        return candidate
    in_models = settings.MODELS_DIR / candidate.name
    if in_models.exists():
        return in_models
    # EXE 内置模型目录
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "data" / "models" / candidate.name
        if bundled.exists():
            return bundled
    return None


def _export_to_onnx(model_path: str) -> Optional[Path]:
    """将 .pt 导出为 .onnx，保存在 MODELS_DIR 下，返回路径。"""
    try:
        from ultralytics import YOLO
        logger.info(f"Exporting {model_path} to ONNX for faster inference...")
        m = YOLO(model_path)
        out = m.export(format="onnx", dynamic=False, simplify=True)
        exported = Path(str(out))
        target = settings.MODELS_DIR / exported.name
        if exported != target:
            import shutil
            shutil.move(str(exported), str(target))
        logger.info(f"ONNX model saved to {target}")
        return target
    except Exception as e:
        logger.warning(f"ONNX export failed, will use PyTorch backend: {e}")
        return None


class YoloDetector(AbstractDetector):
    """
    YOLOv8/v11 目标检测器
    config:
        model: str        - 模型文件路径或名称，如 "yolo11n.pt"
        confidence: float - 置信度阈值，默认 0.5
        iou: float        - NMS IoU 阈值，默认 0.45
        classes: list     - 只检测指定类别名称，如 ["person", "car"]，空列表=检测全部
        device: str       - "auto" | "cpu" | "cuda" | "0"，默认取 settings.YOLO_DEVICE
        detect_interval: float - 检测间隔（秒），默认 1.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model = None
        self._ort_session = None          # onnxruntime 推理会话
        self._class_names: Optional[List[str]] = None
        self._use_onnx: bool = False
        self._device: str = "cpu"

    def initialize(self) -> None:
        model_path = self.config.get("model", settings.YOLO_MODEL)
        requested_device = self.config.get("device", settings.YOLO_DEVICE)
        self._device = _resolve_device(requested_device)

        logger.info(f"Loading YOLO model: {model_path}, device={self._device} (requested={requested_device})")

        # 优先尝试 ONNX Runtime 路径
        if self._try_init_onnx(model_path):
            return

        # 降级到 ultralytics PyTorch
        self._init_pytorch(model_path)

    def _try_init_onnx(self, model_path: str) -> bool:
        """尝试用 onnxruntime 加载模型，成功返回 True。"""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.debug("onnxruntime not available, skipping ONNX path")
            return False

        onnx_path = _onnx_path_for(model_path)
        if onnx_path is None:
            # 仅当源模型是 .pt 时才尝试导出
            if not str(model_path).endswith(".pt"):
                return False
            onnx_path = _export_to_onnx(model_path)
            if onnx_path is None:
                return False

        providers = ["CPUExecutionProvider"]
        if self._device == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.warning("CUDAExecutionProvider not available in onnxruntime, falling back to CPU")
                self._device = "cpu"

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort_session = ort.InferenceSession(
                str(onnx_path), sess_options=sess_options, providers=providers
            )
            # 读取模型元数据中的类别名
            meta = self._ort_session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                raw = ast.literal_eval(meta["names"])
                if isinstance(raw, dict):
                    self._class_names = [raw[i] for i in sorted(raw)]
                else:
                    self._class_names = list(raw)
            self._use_onnx = True
            actual_provider = self._ort_session.get_providers()[0]
            logger.info(f"YOLO ONNX loaded: {onnx_path.name}, provider={actual_provider}")
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"onnxruntime session creation failed: {e}")
            self._ort_session = None
            return False

    def _init_pytorch(self, model_path: str) -> None:
        """降级路径：用 ultralytics PyTorch 加载。"""
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        if self._device and self._device not in ("auto", "cpu"):
            # 仅当 PyTorch 实际能使用 CUDA 时才移到 GPU，避免 CPU-only 构建报错
            try:
                import torch
                if torch.cuda.is_available():
                    self._model.to(self._device)
                else:
                    logger.warning(
                        f"PyTorch CUDA not available (device={self._device}), falling back to CPU"
                    )
                    self._device = "cpu"
            except ImportError:
                logger.warning("torch not importable, keeping model on CPU")
                self._device = "cpu"
        actual = next(self._model.model.parameters()).device
        logger.info(f"YOLO PyTorch loaded: {model_path}, device={actual}")
        self._use_onnx = False
        self._initialized = True

    # ── 推理 ──────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._initialized or (self._model is None and self._ort_session is None):
            self.initialize()

        if self._use_onnx:
            return self._detect_onnx(frame)
        return self._detect_pytorch(frame)

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        """使用 onnxruntime 推理。"""
        import cv2

        confidence_thr = float(self.config.get("confidence", settings.YOLO_CONFIDENCE))
        iou_thr = float(self.config.get("iou", 0.45))
        filter_classes: List[str] = self.config.get("classes", [])

        # 预处理：BGR → RGB, resize to 640, normalize
        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (640, 640))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]  # NCHW

        input_name = self._ort_session.get_inputs()[0].name
        outputs = self._ort_session.run(None, {input_name: inp})
        preds = outputs[0]  # shape: (1, num_classes+4, num_anchors) or (1, N, 6)

        detections: List[Detection] = []

        # 兼容两种常见输出格式
        if preds.ndim == 3 and preds.shape[1] > 6:
            # YOLOv8/v11 格式: (1, 4+nc, num_anchors) — 需要转置
            preds = preds[0].T  # (num_anchors, 4+nc)
            boxes_xywh = preds[:, :4]
            scores = preds[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            confs = scores[np.arange(len(scores)), class_ids]
            mask = confs >= confidence_thr
            boxes_xywh = boxes_xywh[mask]
            class_ids = class_ids[mask]
            confs = confs[mask]
        else:
            # 已经是 (1, N, 6): [x1,y1,x2,y2,conf,cls]
            preds = preds[0]
            mask = preds[:, 4] >= confidence_thr
            preds = preds[mask]
            if len(preds) == 0:
                return detections
            class_ids = preds[:, 5].astype(int)
            confs = preds[:, 4]
            boxes_xywh = preds[:, :4]

        if len(class_ids) == 0:
            return detections

        scale_x, scale_y = w / 640.0, h / 640.0

        # 将坐标统一转为原图尺度下的 [x, y, w, h]（NMSBoxes 需要此格式）
        nms_boxes = []
        for bx in boxes_xywh:
            if bx.shape[0] == 4 and not (bx[0] < 1 and bx[1] < 1):
                cx, cy, bw, bh = float(bx[0]) * scale_x, float(bx[1]) * scale_y, \
                                  float(bx[2]) * scale_x, float(bx[3]) * scale_y
                nms_boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            else:
                x1_ = int(float(bx[0]) * scale_x)
                y1_ = int(float(bx[1]) * scale_y)
                nms_boxes.append([x1_, y1_,
                                   int(float(bx[2]) * scale_x) - x1_,
                                   int(float(bx[3]) * scale_y) - y1_])

        # NMS：去除同类高重叠框，只保留置信度最高的
        indices = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), confidence_thr, iou_thr)
        if len(indices) == 0:
            return detections
        indices = indices.flatten()

        for i in indices:
            cls_id = int(class_ids[i])
            label = (self._class_names[cls_id]
                     if self._class_names and cls_id < len(self._class_names)
                     else str(cls_id))
            if filter_classes and label not in filter_classes:
                continue
            conf = float(confs[i])
            nx, ny, nw, nh = nms_boxes[i]
            detections.append(Detection(
                label=label,
                confidence=conf,
                bbox=[nx, ny, nx + nw, ny + nh],
                metadata={"cls_id": cls_id},
            ))
        return detections

    def _detect_pytorch(self, frame: np.ndarray) -> List[Detection]:
        """使用 ultralytics PyTorch 推理。"""
        confidence = float(self.config.get("confidence", settings.YOLO_CONFIDENCE))
        iou = float(self.config.get("iou", 0.45))
        filter_classes: List[str] = self.config.get("classes", [])

        results = self._model.predict(
            frame,
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                if filter_classes and label not in filter_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    metadata={"cls_id": cls_id},
                ))
        return detections

    def release(self) -> None:
        self._model = None
        self._ort_session = None
        self._initialized = False
        self._use_onnx = False
