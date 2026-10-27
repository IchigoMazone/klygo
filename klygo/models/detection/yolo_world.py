"""Ultralytics YOLO-World detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLOWorldDetect(Detector):
    """Run an Ultralytics YOLO-World model with prompt-defined classes."""

    _NATIVE_OPTIONS = {
        "agnostic_nms", "augment", "batch", "conf", "device", "half", "imgsz",
        "iou", "max_det", "retina_masks", "save", "save_conf", "save_crop",
        "save_frames", "save_json", "save_txt", "show", "show_boxes",
        "show_conf", "show_labels", "tracker", "vid_stride", "visualize",
    }

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
        """Build validated runtime options for YOLO-World native prediction."""
        model_config = dict(metadata["model"])
        post_config = dict(metadata["post"])
        native_post = {
            k: v for k, v in post_config.items()
            if k in self._NATIVE_OPTIONS and k not in ("conf", "iou", "threshold")
        }
        runtime = {**model_config, **native_post}
        runtime.update({
            "conf": post_config.get("threshold", 0.25),
            "iou": post_config.get("iou", 0.7),
        })
        unsupported = set(runtime) - self._NATIVE_OPTIONS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported Ultralytics native options: {names}")
        return runtime

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

        native_post = {
            k: v for k, v in post_config.items()
            if k in self._NATIVE_OPTIONS and k not in ("conf", "iou", "threshold")
        }
        native_kwargs = {**model_config, **native_post}

        results = self.model(
            images,
            conf=post_config.get("threshold", 0.25),
            iou=post_config.get("iou", 0.7),
            verbose=False,
            **native_kwargs,
        )

        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={"prompt": list(self.names.values()), **post_config},
            names=self.names,
        )


__all__ = ["YOLOWorldDetect"]
