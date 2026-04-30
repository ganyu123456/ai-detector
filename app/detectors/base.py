"""检测器抽象基类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class Detection:
    """单次检测结果"""
    label: str
    confidence: float
    bbox: List[int]          # [x1, y1, x2, y2]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AbstractDetector(ABC):
    """所有检测器的统一接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        """加载模型/资源（懒初始化，首次使用前调用）"""
        self._initialized = True

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        对一帧图像执行检测
        :param frame: BGR numpy array (H, W, 3)
        :return: 检测结果列表
        """

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """在帧上绘制检测结果（可选重写）"""
        import cv2
        result = frame.copy()
        h, w = result.shape[:2]

        # 线宽和字号随分辨率自适应（2K 约 4px 线宽，字号 ~0.8）
        thickness = max(2, w // 600)
        font_scale = max(0.5, w / 1920 * 0.75)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)       # 绿框
        text_bg = (0, 200, 0)     # 深绿背景色块
        text_color = (0, 0, 0)    # 黑色文字，高对比

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # 绘制边框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # 标签：label + 百分比置信度
            label_text = f"{det.label} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # 标签背景优先画在框上方，超出顶部则移到框内顶部
            if y1 - th - 8 >= 0:
                bg_y1, bg_y2 = y1 - th - 8, y1
                txt_y = y1 - 4
            else:
                bg_y1, bg_y2 = y1, y1 + th + 8
                txt_y = y1 + th + 2

            # 实心背景色块（不透明，彻底解决文字模糊）
            cv2.rectangle(result, (x1, bg_y1), (x1 + tw + 6, bg_y2), text_bg, cv2.FILLED)
            # 黑色文字叠在色块上
            cv2.putText(result, label_text, (x1 + 3, txt_y),
                        font, font_scale, text_color, max(1, thickness - 1))

        return result

    def release(self) -> None:
        """释放资源"""
        self._initialized = False
