"""
Trình bao bọc mô hình nhận diện đối tượng kiến trúc YOLO (klygo.models.detection.yolo).
"""

from typing import Dict, Any, List, Union, Sequence
import PIL.Image

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLODetect(Detector):
    """Mô hình nhận diện đối tượng thời gian thực YOLO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(
            metadata=metadata,
            flags=("model", "post"),
            **kwargs,
        )
        mod_kw, post_kw = self.parse_config()
        self.model = None
        if self.model_id and self.model_id not in ("custom-detector", "yolo-audit"):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_id)
            except Exception:
                pass

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]] = None,
        **kwargs,
    ) -> List[Detection]:
        mod_kw, post_kw = self.split_kwargs(kwargs)
        raw_list = []
        if self.model is not None:
            conf_thresh = post_kw.get("threshold", 0.25)
            iou_thresh = post_kw.get("iou", 0.7)
            ultra_results = self.model(
                images,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
                **self.filter_kwargs(mod_kw, "torch_dtype", "dtype"),
            )
            raw_list = self.format_results(ultra_results)
        else:
            raw_list = [{"boxes": [], "scores": [], "labels": []} for _ in images]

        return self.build_detections(images, raw_list, **post_kw)
