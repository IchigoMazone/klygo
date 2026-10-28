"""Ultralytics YOLO-World detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

try:
    from ultralytics import YOLOWorld
except ImportError as exc:
    raise ImportError(
        "YOLO-World requires the Ultralytics model extra. "
        "Install it using 'pip install \"klygo[ultralytics]\"'."
    ) from exc

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLOWorldDetect(Detector):
    """Run an Ultralytics YOLO-World model with prompt-defined classes."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)

        from ultralytics import YOLOWorld

        self.model = YOLOWorld(self.model_id)
        self.original_names = dict(self.model.names)
        self.names = dict(self.original_names)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Detection]:

        classes = list(self.original_names.values()) if prompt is None else list(prompt)
        if list(self.model.names.values()) != classes:
            self.model.set_classes(classes)
        self.names = dict(self.model.names)

        model_config = dict(self.metadata["model"])
        post_config = dict(self.metadata["post"])

        threshold = float(kwargs.get("threshold", post_config.get("threshold", 0.25)))
        iou = float(kwargs.get("iou", post_config.get("iou", 0.7)))

        post_opts = utils.exclude_keys(post_config, ["conf", "iou", "threshold"])
        extra_kwargs = utils.exclude_keys(kwargs, ["threshold", "iou", "conf", "prompt"])
        model_args = {**model_config, **post_opts, **extra_kwargs}

        results = self.model(
            images,
            conf=threshold,
            iou=iou,
            verbose=False,
            **utils.exclude_keys(
                model_args,
                ["threshold", "iou"],
            ),
        )

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
