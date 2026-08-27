"""
Trình bao bọc mô hình nhận diện đối tượng kiến trúc LocateAnything (klygo.models.detection.locate_anything).
"""

from typing import Dict, Any, List
import PIL.Image

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class LocateAnythingDetect(Detector):
    """Mô hình nhận diện đối tượng LocateAnything."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(
            metadata=metadata,
            unsupported=("train", "val"),
            **kwargs,
        )

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: List[str],
        model_kwargs: Dict[str, Any],
        processor_kwargs: Dict[str, Any],
        post_kwargs: Dict[str, Any],
    ) -> List[Detection]:
        return []
