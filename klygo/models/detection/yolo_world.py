"""Ultralytics YOLO-World detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLOWorldDetect(Detector):
    """Run an Ultralytics YOLO-World model with prompt-defined classes."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise ImportError(
                "YOLO-World requires the Ultralytics model extra. "
                "Install it using 'pip install \"klygo[ultralytics]\"'."
            ) from exc

        self.model = YOLOWorld(self.model_id)
        self.original_names = dict(self.model.names)
        self.names = dict(self.original_names)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Optional[List[str]] = None,
    ) -> List[Detection]:

        classes = list(self.original_names.values()) if prompt is None else list(prompt)
        if list(self.model.names.values()) != classes:
            self.model.set_classes(classes)
        self.names = dict(self.model.names)

        model_config = dict(self.metadata["model"])
        post_config = dict(self.metadata["post"])

        results = self.model(
            images,
            conf=post_config.get("threshold", 0.25),
            iou=post_config.get("iou", 0.7),
            verbose=False,
            **utils.exclude_keys(
                {**model_config, **post_config},
                ["threshold", "iou"],
            ),
        )

        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={"prompt": list(self.names.values()), **post_config},
            names=self.names,
        )


__all__ = ["YOLOWorldDetect"]
