# `klygo.models.backend.keras` — Keras 3 & KerasHub Backend Guide

```python
from klygo.models.backend import keras
# or via lazy alias:
from klygo.models import backend
adapter = backend.keras  # also backend.kerashub / backend.keras_hub
```

Runtime adapter and `Detector` integration guide for **Keras 3** and **KerasHub** object-detection models, including **RT-DETR / D-FINE** (`dfine_small_coco`, `dfine_medium_coco`, `dfine_large_coco`), **RetinaNet**, and custom `keras.Model` architectures across `KERAS_BACKEND=torch`, `jax`, or `tensorflow`.

---

## 1. Installation & Backend Engine Configuration

```bash
pip install "klygo[keras]" keras-hub
```

Klygo supports running Keras 3 and KerasHub `ObjectDetector` models directly on the **PyTorch runtime (`KERAS_BACKEND="torch"`)** without installing `tensorflow` or `grain`. Configure the engine before importing `keras`:

```python
from klygo import models

models.set_backend("Keras", engine="torch")
```

---

## 2. Adapter Functions (`klygo/models/backend/keras.py`)

When `Detector` dispatches through [`klygo.models.backend.common`](common.md) with `backend="Keras"` (or `"KerasHub"` / `"TensorFlow"`), `klygo.models.backend.keras` provides specialized handling for both standard `keras.Model` instances and `keras_hub.models.ObjectDetector` presets:

- `keras.current_device(model=None) -> str`: Query the active Keras device (`"cuda:0"`, `"gpu:0"`, or `"cpu"`), checking Torch CUDA availability when `KERAS_BACKEND == "torch"`.
- `keras.current_dtype(model=None) -> str`: Return `model.compute_dtype` / `model.dtype` or `keras.config.floatx()`.
- `keras.cast_inputs(inputs, dev=None, dtype=None) -> Any`: Convert `inputs` via `keras.ops.convert_to_tensor()` and cast floating-point tensors to `dtype` while preserving integer tensors.
- `keras.run_inference(model, inputs, **model_kwargs) -> Any`:
  - **For `KerasHub` `ObjectDetector` models** (having `decode_predictions` and `preprocessor`):
    1. Inspects `model.preprocessor.image_converter.image_size` (e.g., `(640, 640)`) and resizes `inputs` via `keras.ops.image.resize` prior to calling `model.preprocessor(x)`. This prevents `DFineImageConverter` (which defaults to `crop_to_aspect_ratio=True`) from center-cropping rectangular images and shifting bounding-box coordinates.
    2. Executes `raw_preds = model(x, training=False, **model_kwargs)`.
    3. Calls `decoded = model.decode_predictions(raw_preds, inputs)` with the original `(H, W)` tensor so normalized coordinates scale back to the exact input image resolution.
    4. Attaches `_bounding_box_format` metadata (e.g., `"yxyx"`) to the returned dictionary.
  - **For standard `keras.Model` instances**: Calls `model(inputs, training=False, **model_kwargs)`.
- `keras.format_results(raw_outputs) -> list[dict[str, Any]]`:
  - Extracts `"boxes"`, `"scores"` (or `"confidence"`), and `"labels"` (or `"classes"`).
  - Respects `num_detections` and filters out padded `-1` detection slots (`score < 0` or `class_id < 0`).
  - Automatically converts `yxyx` (`[ymin, xmin, ymax, xmax]`) bounding boxes from KerasHub into Klygo's standard `xyxy` (`[xmin, ymin, xmax, ymax]`) format.
- `keras.save(model, output_dir) -> None`: Saves via `model.save_to_preset(output_dir)`, `model.save("<output_dir>/model.keras")`, or `model.save_weights("<output_dir>/model.weights.h5")`.
- `keras.synchronize(device=None, value=None)` / `keras.clear_cache()`: Materializes lazy outputs with `keras.ops.convert_to_numpy()` and clears Torch CUDA cache when running on the `"torch"` backend engine without destroying the global Keras session.

---

## 3. Writing a Custom KerasHub `Detector` Subclass (RT-DETR / D-FINE)

```python
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras_hub
from klygo import models, outputs


class KerasHubDFineDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Keras", **kwargs)
        preset = self.model_id or "dfine_small_coco"
        self.model = keras_hub.models.ObjectDetector.from_preset(preset)
        # Map COCO 80-class indices (0..79) to human-readable names
        self.names = {0: "person", 1: "bicycle", 2: "car", 9: "traffic light", 20: "elephant", 77: "teddy bear"}

    def forward(self, images, prompt=None):
        post_cfg = dict(self.metadata.get("post", {}))
        threshold = float(post_cfg.get("threshold", 0.25))
        allowed = set(models.utils.normalize_prompt(prompt)) if prompt else None

        # 1. Stack PIL images as float32 RGB batch [B, H, W, 3]
        batch = np.stack([np.asarray(img.convert("RGB"), dtype=np.float32) for img in images], axis=0)

        # 2. Delegate to Klygo's Keras backend pipeline
        inputs = self.cast_inputs(batch)
        raw_outputs = self.run_inference(inputs)
        formatted = self.format_results(raw_outputs)

        # 3. Filter by confidence threshold and optional prompt
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
        name="D-FINE Small (RT-DETR Family)",
        task="Object-Detection",
        backend="Keras",
        library="keras_hub",
    ),
)
configured = models.configure(
    "dfine_small_coco",
    metadata=schema,
    implementation=models.Detector,
    threshold=0.5,
)
print(configured.model_id, configured.details.backend, configured.post.threshold)
```

See [`examples/integration/multi_backend_workflow.py`](../../../../examples/integration/multi_backend_workflow.py) for the complete evaluation of `dfine_small_coco` on COCO15.
