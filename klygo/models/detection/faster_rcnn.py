"""Torchvision Faster R-CNN object detection."""

from os import PathLike
from typing import Any, Dict

try:
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.transforms.functional import pil_to_tensor
except (ImportError, RuntimeError) as exc:
    raise ImportError(
        "Faster R-CNN requires the Torchvision model extra. "
        "Install it using 'pip install \"klygo[torchvision]\"'."
    ) from exc

from klygo import files, outputs
from klygo.models import utils
from klygo.models.detection.base import Detector
from klygo.models.detection import _torchvision as execution

__all__ = ["FasterRCNNDetect"]


class FasterRCNNDetect(Detector):
    """Run Torchvision Faster R-CNN MobileNetV3-Large FPN."""

    def __init__(self, metadata: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(metadata=metadata, backend="Torchvision", **kwargs)
        model_config = dict(self.metadata.get("model", {}))
        device = model_config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        weights_source = model_config.get("weights", "DEFAULT")

        constructor_options: Dict[str, Any] = {}
        local_weights = isinstance(weights_source, (str, PathLike)) and files.is_file(weights_source)
        if weights_source is None or local_weights:
            constructor_options.update(weights=None, weights_backbone=None)
        else:
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.verify(weights_source)
            constructor_options["weights"] = weights

        self.model = fasterrcnn_mobilenet_v3_large_fpn(**constructor_options)
        if local_weights:
            state = torch.load(weights_source, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        self.model = self.model.to(device).eval()

        categories = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.meta["categories"]
        self.names = {
            class_id: name
            for class_id, name in enumerate(categories)
            if name != "__background__"
        }

    def cast_inputs(self, inputs):
        """Move Torchvision image tensors to the model's device and dtype."""
        return execution.cast_inputs(inputs, self.current_device(), self.current_dtype())

    def run_inference(self, inputs, **model_kwargs):
        """Run the native Torchvision model."""
        return self.model(inputs, **model_kwargs)

    def format_results(self, raw_outputs):
        """Pack Torchvision outputs without framework dispatch."""
        return execution.format_results(raw_outputs)

    def forward(self, images, prompt=None):
        """Run Faster R-CNN and return standardized detection frames."""
        post_config = dict(self.metadata.get("post", {}))
        threshold = float(post_config.get("threshold", 0.5))
        allowed = set(utils.normalize_prompt(prompt)) if prompt else None

        tensors = [
            pil_to_tensor(image.convert("RGB")).float() / 255.0
            for image in images
        ]
        raw_outputs = self.run_inference(self.cast_inputs(tensors))
        formatted = self.format_results(raw_outputs)

        for item in formatted:
            selected = [
                (box, score, class_id, self.names.get(class_id, str(class_id)))
                for box, score, class_id in zip(
                    item["boxes"], item["scores"], item["class_ids"]
                )
                if score >= threshold
                and (
                    allowed is None
                    or self.names.get(class_id, str(class_id)) in allowed
                )
            ]
            item["boxes"] = [box for box, _score, _class_id, _label in selected]
            item["scores"] = [score for _box, score, _class_id, _label in selected]
            item["class_ids"] = [class_id for _box, _score, class_id, _label in selected]
            item["labels"] = [label for _box, _score, _class_id, label in selected]

        return outputs.build_detections(
            images,
            formatted,
            metadata={"prompt": prompt, **post_config},
            names=self.names,
        )
