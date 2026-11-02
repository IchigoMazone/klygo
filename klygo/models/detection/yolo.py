"""Ultralytics YOLO detection wrapper."""

from typing import Any, Dict

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "YOLO requires the Ultralytics model extra. "
        "Install it using 'pip install \"klygo[ultralytics]\"'."
    ) from exc

from klygo import outputs, utils
from klygo.models.detection.base import Detector


class YOLODetect(Detector):
    """Run a fixed-vocabulary Ultralytics YOLO detector."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)
        
        self.model = YOLO(self.model_id)
        self.names = dict(self.model.names)

    def forward(self, images, prompt=None, **kwargs):
        """Run YOLO and return standard detection frames."""
        model_config = dict(self.metadata.get("model", {}))
        post_config = dict(self.metadata.get("post", {}))

        if self.names != self.model.names:
            self.names = dict(self.model.names)

        classes = model_config.pop("classes", None)
        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            unknown = [label for label in requested if label not in self.names.values()]
            if unknown:
                raise ValueError(
                    f"Unknown YOLO classes: {unknown}; use labels from model.names"
                )
            class_ids = utils.invert(self.names)
            classes = [class_ids[label] for label in requested]

        threshold = float(post_config.get("threshold", 0.25))
        iou = float(post_config.get("iou", 0.7))
        options = {
            **model_config,
            **utils.exclude_keys(post_config, ["threshold", "iou", "conf"]),
            "conf": threshold,
            "iou": iou,
            "verbose": False,
        }
        if classes is not None:
            options["classes"] = classes

        results = self.model(images, **options)
        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={
                "prompt": prompt,
                "classes": classes,
                "threshold": threshold,
                "iou": iou,
            },
            names=self.names,
        )


__all__ = ["YOLODetect"]
