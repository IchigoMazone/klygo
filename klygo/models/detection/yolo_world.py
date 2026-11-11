"""Ultralytics YOLO-World detection wrapper."""

from typing import Any, Dict
from contextlib import contextmanager
from klygo.models._frameworks import ultralytics as execution

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
        
        weights = self.metadata.get("model", {}).get("weights")
        self.model = YOLOWorld(weights or self.model_id)
        self.original_names = dict(self.model.names)
        self.names = dict(self.original_names)

    @contextmanager
    def _prediction_session(self, options):
        # Predictor reuse belongs to this wrapper, not the generic task engine.
        with self._prediction_lock:
            execution.optimize_model(self.model)
        yield self.forward

    def format_results(self, raw_outputs):
        """Pack Ultralytics results without framework dispatch."""
        return execution.format_results(raw_outputs)

    def run_inference(self, inputs, **model_kwargs):
        """Use the model's native prediction API."""
        return self.model(inputs, **model_kwargs)

    def forward(self, images, prompt=None, **kwargs):
        """Run YOLO-World and return standard detection frames."""
        model_config = dict(self.metadata.get("model", {}))
        model_config.pop("weights", None)
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
