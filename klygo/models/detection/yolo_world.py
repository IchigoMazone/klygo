"""Ultralytics YOLO-World detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


_YOLO_POST_OPTIONS = {"agnostic_nms", "max_det", "retina_masks", "nms"}


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

    def _native_runtime_options(self, metadata, prompt):
        """Build runtime options for YOLO-World native prediction."""
        model_config = dict(metadata["model"])
        post_config = dict(metadata["post"])
        post_model_opts = {
            k: v for k, v in post_config.items()
            if k in _YOLO_POST_OPTIONS and k not in ("conf", "iou", "threshold")
        }
        runtime = {**model_config, **post_model_opts}
        runtime["conf"] = float(post_config.get("threshold", 0.25))
        runtime["iou"] = float(post_config.get("iou", 0.7))
        return runtime

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Optional[List[str]] = None,
        native: Optional[bool] = None,
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

        post_model_opts = {
            k: v for k, v in post_config.items()
            if k in _YOLO_POST_OPTIONS and k not in ("conf", "iou", "threshold")
        }
        extra_kwargs = utils.exclude_keys(kwargs, ["threshold", "iou", "conf", "prompt", "native"])
        model_args = {**model_config, **post_model_opts, **extra_kwargs}

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
            metadata={"prompt": list(self.names.values()), "threshold": threshold, "iou": iou, "native": bool(native)},
            names=self.names,
        )


__all__ = ["YOLOWorldDetect"]
