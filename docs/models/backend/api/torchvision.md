# `klygo.models.backend.torchvision` — Torchvision Backend Guide

```python
from klygo.models.backend import torchvision
# or via lazy alias:
from klygo.models import backend
adapter = backend.torchvision
```

Runtime adapter and `Detector` integration guide for **Torchvision** object-detection models, including **Faster R-CNN** (`fasterrcnn_mobilenet_v3_large_fpn`, `fasterrcnn_resnet50_fpn_v2`), **RetinaNet**, **FCOS**, and **SSD / SSDLite**.

---

## 1. Installation & Requirements

```bash
pip install "klygo[torchvision]"
```

Select this backend via `backend="Torchvision"` on [`models.Detector`](../../api/Detector.md) or in [`models.details(..., backend="Torchvision")`](../../api/details.md).

---

## 2. Adapter Functions (`klygo/models/backend/torchvision.py`)

Torchvision detection models expect a `list[torch.Tensor]` (where each tensor has shape `[C, H, W]` in `[0.0, 1.0]`) and return a `list[dict[str, torch.Tensor]]` containing `"boxes"`, `"scores"`, and `"labels"`. The Torchvision backend adapter handles this contract automatically:

- `torchvision.current_device(model) -> torch.device`: Detect the device owning `model`'s parameters.
- `torchvision.current_dtype(model) -> torch.dtype`: Detect the dtype of `model`'s parameters.
- `torchvision.inference_context() -> ContextManager`: Return `torch.inference_mode()`.
- `torchvision.cast_inputs(inputs, dev, dtype) -> list[torch.Tensor]`: Move and cast every image tensor in `inputs` via `[image.to(device=dev, dtype=dtype) for image in inputs]`.
- `torchvision.run_inference(model, inputs, **kwargs) -> list[dict[str, torch.Tensor]]`: Execute `model(inputs, **kwargs)`.
- `torchvision.format_results(raw_outputs) -> list[dict[str, Any]]`: Detach and move `"boxes"`, `"scores"`, and `"labels"` to CPU Python lists and populate both `"labels"` and `"class_ids"`.
- `torchvision.save(model, output_dir) -> None`: Save `model.state_dict()` to `<output_dir>/model.pt`.
- `torchvision.reset(model) -> None` / `torchvision.unload(model) -> None`: Move `model` back to CPU (`model.cpu()`).

---

## 3. Writing a Custom Torchvision `Detector` Subclass (Faster R-CNN)

```python
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.transforms.functional import pil_to_tensor
from klygo import models, outputs, utils


class FasterRCNNDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Torchvision", **kwargs)
        model_cfg = dict(self.metadata.get("model", {}))
        device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(device).eval()
        self.names = {
            idx: name
            for idx, name in enumerate(weights.meta["categories"])
            if name != "__background__"
        }

    def forward(self, images, prompt=None):
        model_cfg = dict(self.metadata.get("model", {}))
        post_cfg = dict(self.metadata.get("post", {}))
        threshold = float(post_cfg.get("threshold", 0.25))
        allowed = set(models.utils.normalize_prompt(prompt)) if prompt else None

        # 1. Convert PIL images to [C, H, W] float tensors in [0, 1]
        tensors = [pil_to_tensor(img.convert("RGB")).float() / 255.0 for img in images]

        # 2. Cast tensors, call the native model, then normalize its tensors
        inputs = self.cast_inputs(tensors)
        forward_options = utils.exclude_keys(model_cfg, ["device", "weights"])
        raw_outputs = self.model(inputs, **forward_options)
        formatted = self.format_results(raw_outputs)

        # 3. Filter by confidence threshold and optional class prompt
        for item in formatted:
            boxes, scores, labels, class_ids = [], [], [], []
            for box, score, cid in zip(item["boxes"], item["scores"], item["class_ids"]):
                label = self.names.get(cid, str(cid))
                if score < threshold:
                    continue
                if allowed is not None and label not in allowed:
                    continue
                boxes.append(box)
                scores.append(score)
                labels.append(label)
                class_ids.append(cid)
            item["boxes"] = boxes
            item["scores"] = scores
            item["labels"] = labels
            item["class_ids"] = class_ids

        return outputs.build_detections(
            images,
            formatted,
            metadata={"prompt": prompt, **post_cfg},
            names=self.names,
        )
```

---

## 4. Configuring and Loading

```python
from klygo import models

flags = models.flags(model=False, post=True)
schema = models.metadata(
    flags,
    priority=models.priority(flags, post=("threshold", "iou")),
    details=models.details(
        name="Faster R-CNN MobileNetV3-Large FPN",
        task="Object-Detection",
        backend="Torchvision",
        library="torchvision",
    ),
)
configured = models.configure(
    "fasterrcnn_mobilenet_v3_large_fpn",
    metadata=schema,
    implementation= models.Detector,
    model_device="cpu",
    threshold=0.45,
)
print(configured.model_id, configured.details.backend, configured.post.threshold)
```

See [`examples/integration/multi_backend_workflow.py`](../../../../examples/integration/multi_backend_workflow.py) and [`examples/integration/torchvision_coco15.py`](../../../../examples/integration/torchvision_coco15.py) for complete real-image workflows.
