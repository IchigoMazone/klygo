"""
Trình bao bọc mô hình nhận diện đối tượng kiến trúc YOLO (klygo.models.detection.yolo).
"""

from typing import Dict, Any, List
import PIL.Image

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLODetect(Detector):
    """Mô hình nhận diện đối tượng thời gian thực YOLO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        # YOLO hỗ trợ đầy đủ nên không khóa tính năng nào
        super().__init__(metadata=metadata, **kwargs)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: List[str],
        model_kwargs: Dict[str, Any],
        processor_kwargs: Dict[str, Any],
        post_kwargs: Dict[str, Any],
    ) -> List[Detection]:
        return []
