"""
Trình bao bọc mô hình nhận diện đối tượng kiến trúc YOLO (klygo.models.detection.yolo).
"""

from typing import Dict, Any, List, Union
import PIL.Image

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLODetect(Detector):
    """Mô hình nhận diện đối tượng thời gian thực YOLO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, flags=("model", "post"), **kwargs)
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
            for res in ultra_results:
                boxes_data, scores_data, labels_data = [], [], []
                if res.boxes is not None:
                    for b in res.boxes:
                        boxes_data.append(b.xyxy[0].tolist())
                        scores_data.append(float(b.conf[0].item()))
                        cls_id = int(b.cls[0].item())
                        labels_data.append(res.names.get(cls_id, str(cls_id)))
                raw_list.append({"boxes": boxes_data, "scores": scores_data, "labels": labels_data})
        else:
            for img in images:
                raw_list.append({"boxes": [], "scores": [], "labels": []})

        return self.build_detections(images, raw_list, **post_kw)
