# `klygo.models` Documentation

`klygo.models` provides a unified, multi-backend AI model orchestration layer for configuring metadata schemas, resolving local or registry models, managing hardware lifecycles, and executing standardized object-detection inference across **Ultralytics**, **Hugging Face / PyTorch**, **Torchvision**, and **Keras 3 / KerasHub**.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/models/Document_Models_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/models/`, 1-to-1 with `klygo/models/` submodules):
- [`Document_Models_Overview.ipynb`](../../tutorials/models/Document_Models_Overview.ipynb)
- [`Document_Models_Base.ipynb`](../../tutorials/models/Document_Models_Base.ipynb) (`klygo/models/base.py` & `klygo/models/detection/base.py`)
- [`Document_Models_Load.ipynb`](../../tutorials/models/Document_Models_Load.ipynb) (`klygo/models/load.py`)
- [`Document_Models_Metadata.ipynb`](../../tutorials/models/Document_Models_Metadata.ipynb) (`klygo/models/metadata.py`)
- [`Document_Models_Utils.ipynb`](../../tutorials/models/Document_Models_Utils.ipynb) (`klygo/models/utils.py` & `klygo/models/errors.py`)

---

## 1. 3-Tier Architecture & Multi-Backend Matrix

<details open>
<summary><b>3-tier class hierarchy and 4 supported backend families</b> <i>(Click to expand or collapse)</i></summary>

`klygo.models` separates metadata/state management, task-level inference orchestration, and framework-specific runtime execution into three clean layers:

1. **Layer 1 — Abstract Contract ([`BaseModel`](api/BaseModel.md) in `base.py`)**:
   Manages lifecycle state (`LOADING`, `READY`, `UNLOADED`), `metadata`, `details`, `flags`, call-local runtime parameter overrides (`_use_metadata`), Rich terminal inspection (`info()`), and post-unload access guards (`InvalidStateError`).
2. **Layer 2 — Task Engine ([`Detector`](api/Detector.md) in `detection/base.py` & [`backend.common`](backend/README.md))**:
   Implements the complete object-detection pipeline (`predict()`, streaming video/image generators via `stream=True`, coordinate restoration when paired with `klygo.preprocessing`, `warmup()`, `benchmark()`, `save()`, `reset()`, and `unload()`) and delegates low-level tensor/device operations to [`klygo.models.backend.common`](backend/api/common.md).
3. **Layer 3 — Concrete & Application Wrappers (`detection/*.py` & custom subclasses)**:
   Built-in registry wrappers (`YOLODetect`, `YOLOWorldDetect`, `GroundingDinoDetect`) or application-owned [`Detector`](api/Detector.md) subclasses implement `forward(images, prompt=None)` using `self.cast_inputs()`, `self.run_inference()`, and `self.format_results()` (or by calling [`klygo.models.backend.common`](backend/api/common.md) directly).

See the dedicated [**`klygo.models.backend` & `common` Dispatchers Guide**](backend/README.md) and [`common` API Reference](backend/api/common.md) for all 15 unified backend functions (`backend_family`, `current_device`, `current_dtype`, `is_accelerator_available`, `inference_context`, `synchronize`, `clear_cache`, `cast_inputs`, `run_inference`, `get_output_device`, `sync_device`, `format_results`, `save`, `reset`, `unload`).

### Supported Backend Families

| Backend Family | `backend` Identifier | Upstream Libraries | Representative Models | Built-in / Adapter Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Ultralytics** | `"Ultralytics"` | `ultralytics`, `torch` | YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO-World v2 | Built-in `YOLODetect` & `YOLOWorldDetect`, `.pt` direct file loading, open-vocabulary `set_classes` |
| **Hugging Face / PyTorch** | `"Hugging Face"` / `"PyTorch"` | `transformers`, `torch` | Grounding DINO, DETR, RT-DETR, custom `nn.Module` | Built-in `GroundingDinoDetect`, AMP `autocast`, `BatchFeature` float casting, multi-GPU `sync_device` |
| **Torchvision** | `"Torchvision"` | `torchvision`, `torch` | Faster R-CNN (`fasterrcnn_mobilenet_v3_large_fpn`), RetinaNet, FCOS, SSD | List-of-tensor `cast_inputs`, `run_inference`, and `format_results` (`boxes`, `scores`, `labels`) |
| **Keras 3 / KerasHub** | `"Keras"` / `"KerasHub"` | `keras`, `keras-hub` | D-FINE / RT-DETR (`dfine_small_coco`), RetinaNet, custom `keras.Model` | Native `KERAS_BACKEND=torch`/`jax`/`tensorflow` support, `ObjectDetector` aspect-preserving resize & `decode_predictions`, automatic `yxyx` $\rightarrow$ `xyxy` conversion |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (18 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Load a built-in registered model by name** | [`load()`](api/load.md) | `detector = models.load("yolov8n")` |
| **Load a local Ultralytics `.pt` checkpoint directly** | [`load()`](api/load.md) | `detector = models.load("weights/yolov8n.pt", threshold=0.35)` |
| **Load a saved Klygo model directory (`klygo.json`)** | [`load()`](api/load.md) | `detector = models.load("artifacts/saved_detector")` |
| **Load a model from a persisted `klygo.config` file** | [`load()`](api/load.md) | `detector = models.load(config.load("model.yaml"))` |
| **Wrap an in-memory PyTorch `nn.Module`** | [`load()`](api/load.md) | `wrapper = models.load(torch_module)` |
| **Declare parameter groups and runtime permissions** | [`flags()`](api/flags.md) | `f = models.flags(model=False, processor=True, post=True)` |
| **Allow unprefixed parameter names (`threshold=0.4`)** | [`priority()`](api/priority.md) | `p = models.priority(f, post=("threshold", "iou"))` |
| **Attach descriptive model provenance** | [`details()`](api/details.md) | `d = models.details(name="FasterRCNN", task="Detection", backend="Torchvision", library="torchvision")` |
| **Combine schema flags, priority, and details** | [`metadata()`](api/metadata.md) | `schema = models.metadata(f, priority=p, details=d)` |
| **Bind `model_id`, class, and parameter values** | [`configure()`](api/configure.md) | `cfg = models.configure("coco-v1", metadata=schema, implementation=MyDetector, threshold=0.5)` |
| **Build a custom multi-backend detector** | [`Detector`](api/Detector.md) | `class MyDetector(models.Detector): def forward(self, images, prompt=None): ...` |
| **Pack formatted backend dicts into `Detection` frames** | [`outputs.build_detections()`](../outputs/api/build_detections.md) | `outputs.build_detections(images, self.format_results(raw), names=self.names)` |
| **Run batch detection on images, folders, or videos** | [`Detector.predict()`](api/Detector.md) | `dets = detector.predict("images/", threshold=0.4)` |
| **Stream large videos lazily without filling RAM** | [`Detector.predict()`](api/Detector.md) | `for frame_det in detector.predict("traffic.mp4", stream=True, step=2): ...` |
| **Measure model latency (ms) and throughput (FPS)** | [`Detector.benchmark()`](api/Detector.md) | `stats = detector.benchmark("sample.jpg", num_runs=50, warmup=5)` |
| **Export a self-contained model bundle to disk** | [`Detector.save()`](api/Detector.md) | `detector.save("exported_model/")` |
| **Release GPU/CPU memory and lock the instance** | [`Detector.unload()`](api/Detector.md) | `detector.unload()  # Subsequent predict() raises InvalidStateError` |
| **Set or inspect the default backend environment** | [`set_backend()`](api/set_backend.md) / [`get_backend()`](api/get_backend.md) | `models.set_backend("Keras", engine="torch")` |
| **Silence noisy framework logs and warnings** | [`suppress_warnings()`](api/suppress_warnings.md) / [`suppress_ai_warnings()`](api/suppress_ai_warnings.md) | `with models.suppress_warnings(): ...` |

</details>

---

## 3. Public APIs Reference (15 Exports across 5 Submodules)

<details open>
<summary><b>Base contract and detection engine — <code>base.py</code> &amp; <code>detection/base.py</code> (3 exports)</b> <i>(Click to expand)</i></summary>

| API | Description | Example |
| :--- | :--- | :--- |
| [`base`](api/base.md) | Submodule namespace exposing [`BaseModel`](api/BaseModel.md). | [Run](../../examples/models/base.py) |
| [`BaseModel`](api/BaseModel.md) | Abstract base wrapper managing `metadata`, `details`, `flags`, `state`, and post-unload access guards. | [Run](../../examples/models/BaseModel.py) |
| [`Detector`](api/Detector.md) | Base object-detection engine providing `predict()`, `stream=True`, `benchmark()`, `warmup()`, `save()`, `unload()`, and backend dispatch helpers. | [Run](../../examples/models/Detector.py) |

</details>

<details open>
<summary><b>Model resolution, configuration, and backend selection — <code>load.py</code> (4 functions)</b> <i>(Click to expand)</i></summary>

| API | Description | Example |
| :--- | :--- | :--- |
| [`load(model, **kwargs)`](api/load.md) | Resolve and instantiate a model from a metadata mapping, `Config`, offline directory, `.pt` file, registry name, or in-memory module. | [Run](../../examples/models/load.py) |
| [`configure(model_id, ...)`](api/configure.md) | Bind a `model_id`, `implementation` class, and parameter overrides to a validated metadata schema. | [Run](../../examples/models/configure.py) |
| [`set_backend(backend, engine=None)`](api/set_backend.md) | Configure `KLYGO_BACKEND` and optional `KERAS_BACKEND` environment variables. | [Run](../../examples/models/set_backend.py) |
| [`get_backend()`](api/get_backend.md) | Return the active `KLYGO_BACKEND` environment setting or `"auto"`. | [Run](../../examples/models/get_backend.py) |

</details>

<details open>
<summary><b>Metadata schema builders — <code>metadata.py</code> (4 functions)</b> <i>(Click to expand)</i></summary>

| API | Description | Example |
| :--- | :--- | :--- |
| [`metadata(flags, priority=None, details=None)`](api/metadata.md) | Combine parameter-group `flags`, `priority` aliases, and descriptive `details` into a schema `Box`. | [Run](../../examples/models/metadata.py) |
| [`flags(*groups, **permissions)`](api/flags.md) | Declare parameter groups (`model`, `processor`, `post`) and whether `predict()` may override them at runtime. | [Run](../../examples/models/flags.py) |
| [`priority(flags, **parameters)`](api/priority.md) | Register unprefixed parameter names (such as `threshold` or `iou`) mapped to their owning group. | [Run](../../examples/models/priority.py) |
| [`details(*, name, task, backend, library, ...)`](api/details.md) | Build validated descriptive model metadata (`name`, `task`, `backend`, `library`, plus optional attributes). | [Run](../../examples/models/details.py) |

</details>

<details open>
<summary><b>Utilities and runtime exceptions — <code>utils.py</code> &amp; <code>errors.py</code> (4 exports)</b> <i>(Click to expand)</i></summary>

| API | Description | Example |
| :--- | :--- | :--- |
| [`utils`](api/utils.md) | Submodule namespace with metadata normalization, parameter resolution, image/stream resolution, and warning helpers. | [Run](../../examples/models/utils.py) |
| [`errors`](api/errors.md) | Submodule namespace defining `ModelFrameworkError`, `InvalidStateError`, `InvalidConfigError`, `InvalidDeviceError`, and `InvalidDtypeError`. | [Run](../../examples/models/errors.py) |
| [`suppress_warnings(func=None)`](api/suppress_warnings.md) | Context manager and decorator that silences Python and AI library warnings within a scoped block. | [Run](../../examples/models/suppress_warnings.py) |
| [`suppress_ai_warnings()`](api/suppress_ai_warnings.md) | Configure quiet logging and environment flags for Hugging Face, Torch, and Accelerate without importing them. | [Run](../../examples/models/suppress_ai_warnings.py) |

</details>

---

## 4. Custom Multi-Backend Detector Workflow

Any framework detector (Torchvision Faster R-CNN, KerasHub RT-DETR / D-FINE, Ultralytics YOLO, or custom PyTorch models) integrates into Klygo by subclassing [`models.Detector`](api/Detector.md) and delegating tensor casting, execution, and output formatting to the built-in backend helpers:

```python
from klygo import models, outputs


class CustomBackendDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Torchvision", **kwargs)
        # Attach self.model (e.g., fasterrcnn_mobilenet_v3_large_fpn or KerasHub ObjectDetector)
        self.names = {1: "person", 2: "bicycle", 3: "car"}

    def forward(self, images, prompt=None):
        threshold = float(self.metadata.get("post", {}).get("threshold", 0.25))
        # Use self.cast_inputs(), self.run_inference(), and self.format_results()
        # to automatically dispatch to the active backend adapter.
        raw_predictions = [
            {"boxes": [[12.0, 18.0, 140.0, 220.0]], "scores": [0.94], "labels": [1], "class_ids": [1]}
            for _ in images
        ]
        return outputs.build_detections(
            images,
            raw_predictions,
            metadata={"threshold": threshold, "prompt": prompt},
            names=self.names,
        )


schema = models.metadata(
    models.flags(model=False, post=True),
    priority=models.priority(models.flags(model=False, post=True), post=("threshold",)),
    details=models.details(
        name="CustomBackendDetector",
        task="Object-Detection",
        backend="Torchvision",
        library="torchvision",
    ),
)
configured = models.configure(
    "custom-fasterrcnn",
    metadata=schema,
    implementation=CustomBackendDetector,
    threshold=0.4,
)
detector = models.load(configured)
```

For a complete, real-image benchmark running **Ultralytics YOLOv8**, **Ultralytics YOLO-World**, **Torchvision Faster R-CNN**, and **KerasHub RT-DETR (D-FINE)** side by side on COCO15, see [`examples/integration/multi_backend_workflow.py`](../../examples/integration/multi_backend_workflow.py) and the [integration workflows guide](../integration/README.md).
