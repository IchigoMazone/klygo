"""Ultralytics YOLO-World detection wrapper."""

from typing import Any, Dict

try:
    from ultralytics import YOLOWorld
except ImportError as exc:
    raise ImportError(
        "YOLO-World requires the Ultralytics model extra. "
        "Install it using 'pip install \"klygo[ultralytics]\"'."
    ) from exc

from klygo import outputs, utils
from klygo.models.detection.base import Detector


class YOLOWorldDetect(Detector):
    """Run an Ultralytics YOLO-World model with prompt-defined classes."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)
        
        self.model = YOLOWorld(self.model_id)
        self.original_names = dict(self.model.names)
        self.names = dict(self.original_names)

    def forward(self, images, prompt=None, **kwargs):
        """Run YOLO-World and return standard detection frames."""
        model_config = dict(self.metadata.get("model", {}))
        post_config = dict(self.metadata.get("post", {}))

        classes = list(self.original_names.values()) if prompt is None else list(prompt)
        if list(self.model.names.values()) != classes:
            self.model.set_classes(classes)
        self.names = dict(self.model.names)

        threshold = float(post_config.get("threshold", 0.25))
        iou = float(post_config.get("iou", 0.7))
        options = {
            **model_config,
            **utils.exclude_keys(post_config, ["threshold", "iou", "conf"]),
            "conf": threshold,
            "iou": iou,
            "verbose": False,
        }

        results = self.model(images, **options)
        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={
                "prompt": list(self.names.values()),
                "threshold": threshold,
                "iou": iou,
            },
            names=self.names,
        )


__all__ = ["YOLOWorldDetect"]
