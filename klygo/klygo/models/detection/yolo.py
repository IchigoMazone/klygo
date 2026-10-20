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
        super().__init__(
            metadata=metadata,
            **kwargs,
        )
        self.model = None
        self._prompt_classes = None
        if self.model_id and self.model_id not in ("custom-detector", "yolo-audit"):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_id)
            except ImportError as exc:
                raise ImportError(
                    "YOLO requires the Ultralytics model extra. "
                    "Install it using 'pip install \"klygo[ultralytics]\"'."
                ) from exc

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]] = None,
    ) -> List[Detection]:
        model_config = dict(self.metadata["model"])
        post_config = dict(self.metadata["post"])
        raw_list = []
        if self.model is not None:
            prompt_classes = (prompt,) if isinstance(prompt, str) else tuple(prompt or [])
            if prompt_classes and prompt_classes != self._prompt_classes:
                set_classes = getattr(self.model, "set_classes", None)
                if not callable(set_classes):
                    raise ValueError(
                        "This YOLO checkpoint is closed-set and cannot detect arbitrary "
                        "prompt classes. Use a YOLO-World checkpoint such as "
                        "'yolov8s-worldv2.pt'."
                    )
                set_classes(list(prompt_classes))
                self._prompt_classes = prompt_classes

            conf_thresh = post_config.get("threshold", 0.25)
            iou_thresh = post_config.get("iou", 0.7)
            ultra_results = self.model(
                images,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
                **self.filter_kwargs(model_config, "torch_dtype", "dtype"),
            )
            raw_list = self.format_results(ultra_results)
        else:
            raw_list = [{"boxes": [], "scores": [], "labels": []} for _ in images]

        return self.build_detections(images, raw_list, prompt=prompt, **post_config)
