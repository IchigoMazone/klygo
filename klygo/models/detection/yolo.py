"""Ultralytics YOLO detection wrapper."""

from typing import Any, Dict, List, Optional

import PIL.Image

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "YOLO requires the Ultralytics model extra. "
        "Install it using 'pip install \"klygo[ultralytics]\"'."
    ) from exc

from klygo import outputs, utils
from klygo.models.backend import ultralytics as ultralytics_backend
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class YOLODetect(Detector):
    """Run a fixed-vocabulary Ultralytics YOLO detector."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, **kwargs)

        from ultralytics import YOLO

        self.model = YOLO(self.model_id)
        self.names = dict(self.model.names)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Detection]:

        model_config = dict(self.metadata["model"])
        post_config = dict(self.metadata["post"])
        if self.names != self.model.names:
            self.names = dict(self.model.names)
        classes = model_config.get("classes")

        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            unknown = [label for label in requested if label not in self.names.values()]
            if unknown:
                raise ValueError(f"Unknown YOLO classes: {unknown}; use labels from model.names")
            class_ids = utils.invert(self.names)
            classes = [class_ids[name] for name in requested]

        threshold = float(kwargs.get("threshold", post_config.get("threshold", 0.25)))
        iou = float(kwargs.get("iou", post_config.get("iou", 0.7)))

        post_opts = utils.exclude_keys(post_config, ["conf", "iou", "threshold"])
        extra_kwargs = utils.exclude_keys(kwargs, ["threshold", "iou", "conf", "classes", "prompt"])
        model_args = {**model_config, **post_opts, **extra_kwargs}

        predict_options = {
            "conf": threshold,
            "iou": iou,
            "classes": classes,
            "verbose": False,
            **utils.exclude_keys(model_args, ["threshold", "iou", "classes"]),
        }
        prepared_options = getattr(self, "_prepared_predict_options", None)
        results = None
        if prepared_options == predict_options:
            results = ultralytics_backend.predict_prepared(
                self.model,
                images,
                options=predict_options,
            )
        if results is None:
            results = self.model(images, **predict_options)
            self._prepared_predict_options = predict_options

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

    def _stream_state_isolation_required(self) -> bool:
        """Fixed-vocabulary YOLO does not mutate backend state between frames."""
        return False

    def _adopt_forward_detections(self) -> bool:
        """YOLO forward creates fresh unowned frames for each call."""
        return True


__all__ = ["YOLODetect"]
