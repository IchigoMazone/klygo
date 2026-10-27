"""Ultralytics YOLO detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLODetect(Detector):
    """Run a fixed-vocabulary Ultralytics YOLO detector."""

    _NATIVE_OPTIONS = {
        "agnostic_nms", "augment", "batch", "classes", "conf", "device", "half",
        "imgsz", "iou", "max_det", "nms", "retina_masks", "save", "save_conf",
        "save_crop", "save_frames", "save_json", "save_txt", "show",
        "show_boxes", "show_conf", "show_labels", "tracker", "vid_stride", "visualize",
    }

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

    def _native_runtime_options(self, metadata, prompt):
        """Build validated runtime options for Ultralytics native prediction."""
        model_config = dict(metadata["model"])
        post_config = dict(metadata["post"])
        classes = model_config.pop("classes", None)
        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            unknown = [label for label in requested if label not in self.names.values()]
            if unknown:
                raise ValueError(f"Unknown YOLO classes: {unknown}; use labels from model.names")
            class_ids = utils.invert(self.names)
            classes = [class_ids[name] for name in requested]
        native_post = {
            k: v for k, v in post_config.items()
            if k in self._NATIVE_OPTIONS and k not in ("conf", "iou", "threshold")
        }
        runtime = {**model_config, **native_post}
        runtime.update({
            "conf": post_config.get("threshold", 0.25),
            "iou": post_config.get("iou", 0.7),
        })
        if classes is not None:
            runtime["classes"] = classes
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

        native_post = {
            k: v for k, v in post_config.items()
            if k in self._NATIVE_OPTIONS and k not in ("conf", "iou", "threshold")
        }
        native_kwargs = {**model_config, **native_post}

        results = self.model(
            images,
            conf=post_config.get("threshold", 0.25),
            iou=post_config.get("iou", 0.7),
            classes=classes,
            verbose=False,
            **utils.exclude_keys(
                native_kwargs,
                ["classes"],
            ),
        )

        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={"prompt": prompt, "classes": classes, **post_config},
            names=self.names,
        )


__all__ = ["YOLODetect"]
