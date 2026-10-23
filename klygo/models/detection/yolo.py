"""Ultralytics YOLO detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLODetect(Detector):
    """Run a fixed-vocabulary Ultralytics YOLO detector."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO requires the Ultralytics model extra. "
                "Install it using 'pip install \"klygo[ultralytics]\"'."
            ) from exc

        self.model = YOLO(self.model_id)
        self.names = dict(self.model.names)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Optional[List[str]] = None,
    ) -> List[Detection]:
        
        model_config = dict(self.metadata["model"])
        post_config = dict(self.metadata["post"])
        self.names = dict(self.model.names)
        classes = model_config.get("classes")

        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            unknown = [label for label in requested if label not in self.names.values()]
            if unknown:
                raise ValueError(f"Unknown YOLO classes: {unknown}; use labels from model.names")
            class_ids = utils.invert(self.names)
            classes = [class_ids[name] for name in requested]

        results = self.model(
            images,
            conf=post_config.get("threshold", 0.25),
            iou=post_config.get("iou", 0.7),
            classes=classes,
            verbose=False,
            **utils.exclude_keys(
                {**model_config, **post_config},
                ["threshold", "iou", "classes"],
            ),
        )

        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={"prompt": prompt, "classes": classes, **post_config},
            names=self.names,
        )


__all__ = ["YOLODetect"]
