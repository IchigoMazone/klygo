# `klygo.models.backend.ultralytics` — Ultralytics Backend Guide

```python
from klygo.models.backend import ultralytics
# or via lazy alias:
from klygo.models import backend
adapter = backend.ultralytics  # also backend.ul
```

Runtime adapter and `Detector` integration guide for **Ultralytics** models, including fixed-vocabulary **YOLOv8 / YOLOv9 / YOLOv10 / YOLO11** (`YOLODetect`) and open-vocabulary **YOLO-World v2** (`YOLOWorldDetect`).

---

## 1. Installation & Requirements

```bash
pip install "klygo[ultralytics]"
```

 select this backend via `backend="Ultralytics"` on [`models.Detector`](../../api/Detector.md) or in [`models.details(..., backend="Ultralytics")`](../../api/details.md).

---

## 2. Adapter Functions (`klygo/models/backend/ultralytics.py`)

All functions are automatically invoked when `Detector` dispatches through [`klygo.models.backend.common`](common.md) with `backend="Ultralytics"`:

- `ultralytics.current_device(model) -> torch.device`: Return the PyTorch device hosting the Ultralytics model parameters.
- `ultralytics.current_dtype(model) -> torch.dtype`: Return the parameter dtype (`torch.float32`, `torch.float16`).
- `ultralytics.inference_context() -> ContextManager`: Return `torch.inference_mode()`.
- `ultralytics.is_accelerator_available(device=None) -> bool`: Check whether CUDA is available for the active device.
- `ultralytics.synchronize(device=None, value=None) -> None`: Synchronize CUDA streams via `torch.cuda.synchronize()`.
- `ultralytics.clear_cache() -> None`: Release unused PyTorch CUDA cache blocks.
- `ultralytics.format_results(ultra_results) -> list[dict[str, Any]]`: Extract `b.xyxy[0]`, `b.conf[0]`, `b.cls[0]`, and `res.names[cls_id]` from Ultralytics `Results` objects into Klygo's standard `[{'boxes', 'scores', 'labels', 'class_ids'}]` format.
- `ultralytics.save(model, output_dir) -> None`: Save weights to `output_dir` via `model.save()`.
- `ultralytics.reset(model) -> None` / `ultralytics.unload(model) -> None`: Move the Ultralytics model back to CPU (`model.cpu()`).

---

## 3. Writing a Custom Ultralytics `Detector` Subclass

### A. Fixed-Vocabulary YOLO (`YOLOv8` / `YOLO11`)

```python
from ultralytics import YOLO
from klygo import models, outputs, utils


class CustomYOLODetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Ultralytics", **kwargs)
        self.model = YOLO(self.model_id)
        self.names = dict(self.model.names)

    def forward(self, images, prompt=None, **kwargs):
        model_config = dict(self.metadata.get("model", {}))
        post_config = dict(self.metadata.get("post", {}))
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
            metadata={"prompt": prompt, "threshold": threshold, "iou": iou},
            names=self.names,
        )
```

### B. Open-Vocabulary YOLO-World (`yolov8s-worldv2.pt`)

```python
from ultralytics import YOLOWorld
from klygo import models, outputs, utils


class CustomYOLOWorldDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Ultralytics", **kwargs)
        self.model = YOLOWorld(self.model_id)
        self.original_names = dict(self.model.names)
        self.names = dict(self.original_names)

    def forward(self, images, prompt=None, **kwargs):
        model_config = dict(self.metadata.get("model", {}))
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
            metadata={"prompt": list(self.names.values()), "threshold": threshold, "iou": iou},
            names=self.names,
        )
```

`predict()` first routes registered priority names and prefixed parameters into
the active metadata groups. The wrapper deliberately shows those groups, the
native model call, result formatting, and `Detection` construction so each step
can be customized without replacing the prediction engine.

---

## 4. Configuring and Loading

```python
from klygo import models

flags = models.flags(model=False, post=True)
schema = models.metadata(
    flags,
    priority=models.priority(flags, post=("threshold", "iou")),
    details=models.details(
        name="YOLOv8n",
        task="Object-Detection",
        backend="Ultralytics",
        library="ultralytics",
    ),
)
configured = models.configure(
    "yolov8n.pt",
    metadata=schema,
    implementation="klygo.models.detection.yolo.YOLODetect",
    threshold=0.35,
    iou=0.5,
)
# Pass `configured` to models.load(configured) when weights are present.
print(configured.model_id, configured.details.backend, configured.post.threshold)
```

See [`examples/integration/multi_backend_workflow.py`](../../../../examples/integration/multi_backend_workflow.py) for a full evaluation of both `YOLODetect` and `YOLOWorldDetect` on COCO15.
