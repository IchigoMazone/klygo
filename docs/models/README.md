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

Focused guides:

- [Persistent detector configuration and atomic output replacement](detector-output-refactor.md)
- [NVIDIA Locate Anything adapter](locate-anything.md)

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
   Built-in registry wrappers (`YOLODetect`, `YOLOWorldDetect`, `GroundingDinoDetect`, `LocateAnythingDetect`) or application-owned [`Detector`](api/Detector.md) subclasses implement `forward(images, prompt=None)` using `self.cast_inputs()`, `self.run_inference()`, and `self.format_results()` (or by calling [`klygo.models.backend.common`](backend/api/common.md) directly).

See the dedicated [**`klygo.models.backend` & `common` Dispatchers Guide**](backend/README.md) and [`common` API Reference](backend/api/common.md) for all 15 unified backend functions (`backend_family`, `current_device`, `current_dtype`, `is_accelerator_available`, `inference_context`, `synchronize`, `clear_cache`, `cast_inputs`, `run_inference`, `get_output_device`, `sync_device`, `format_results`, `save`, `reset`, `unload`).

### Supported Backend Families

| Backend Family | Dedicated Guide | `backend` Identifier | Upstream Libraries | Representative Models | Built-in / Adapter Capabilities |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ultralytics** | [Ultralytics Guide](backend/api/ultralytics.md) | `"Ultralytics"` | `ultralytics`, `torch` | YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO-World v2 | Built-in `YOLODetect` & `YOLOWorldDetect`, `.pt` direct file loading, open-vocabulary `set_classes` |
| **Hugging Face / PyTorch** | [Hugging Face Guide](backend/api/huggingface.md) | `"Hugging Face"` / `"PyTorch"` | `transformers`, `torch` | Grounding DINO, Locate Anything, DETR, RT-DETR, custom `nn.Module` | Built-in `GroundingDinoDetect` and optional CUDA-only `LocateAnythingDetect`, AMP `autocast`, `BatchFeature` float casting, multi-GPU `sync_device` |
| **Torchvision** | [Torchvision Guide](backend/api/torchvision.md) | `"Torchvision"` | `torchvision`, `torch` | Faster R-CNN (`fasterrcnn_mobilenet_v3_large_fpn`), RetinaNet, FCOS, SSD | List-of-tensor `cast_inputs`, `run_inference`, and `format_results` (`boxes`, `scores`, `labels`) |
| **Keras 3 / KerasHub** | [Keras & KerasHub Guide](backend/api/keras.md) | `"Keras"` / `"KerasHub"` | `keras`, `keras-hub` | D-FINE / RT-DETR (`dfine_small_coco`), RetinaNet, custom `keras.Model` | Native `KERAS_BACKEND=torch`/`jax`/`tensorflow` support, `ObjectDetector` aspect-preserving resize & `decode_predictions`, automatic `yxyx` $\rightarrow$ `xyxy` conversion |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (18 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Load a built-in registered model by name** | [`load()`](api/load.md) | `detector = models.load("yolov8n.pt")` |
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
| **Run batch detection on images, folders, or videos** | [`Detector.predict()`](api/Detector.md) | `dets = detector.predict("images/", prompt=["person"], threshold=0.4)` |
| **Inspect remembered effective inference settings** | [`Detector.predict_config`](api/Detector.md) | `cfg = detector.predict_config  # {'prompt': ['person'], 'threshold': 0.4}` |
| **Inspect hardware device and model data type** | [`Detector.device`](api/Detector.md) / [`dtype`](api/Detector.md) | `print(detector.device, detector.dtype)` |
| **Stream large videos lazily without filling RAM** | [`Detector.predict()`](api/Detector.md) | `source = media.load("traffic.mp4", stream=True); results = detector.predict(source)` |
| **Overlap parallel image decoding with inference** | [`Detector.predict()`](api/Detector.md) | `results = detector.predict(source, batch=8, workers=2, prefetch=2)` |
| **Measure model latency (ms) and throughput in isolation** | [`Detector.benchmark()`](api/Detector.md) | `stats = detector.benchmark("sample.jpg", iterations=10, warmup=3)` |
| **Clear remembered prompt or restore defaults** | [`Detector.predict()`](api/Detector.md) / [`reset()`](api/Detector.md) | `detector.predict(img, prompt=None)` or `detector.reset()` |
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

## 4. How to Write a Custom `Detector` Subclass (All 4 Backends)

Every model in Klygo follows a standardized **4-step pattern** when subclassing [`models.Detector`](api/Detector.md):

1. **`__init__(self, metadata=None, **kwargs)`**: Call `super().__init__(metadata=metadata, backend="...", **kwargs)`, instantiate the underlying framework model into `self.model` (and `self.processor` if needed), and set `self.names` (`dict[int, str]`).
2. **`forward(self, images, prompt=None) -> list[Detection]`**:
   - Read active runtime settings from `self.metadata["model"]`, `self.metadata.get("processor", {})`, and `self.metadata["post"]`.
   - Preprocess `images` (`list[PIL.Image.Image]`) and cast tensors with `self.cast_inputs(...)`.
   - Execute the forward pass with `self.run_inference(...)`.
   - Normalize outputs with `self.format_results(...)` and pack into `Detection` frames via [`outputs.build_detections()`](../outputs/api/build_detections.md).
3. **Define schema (`flags`, `priority`, `details`, `metadata`)**: Declare which parameter groups (`model`, `processor`, `post`) are locked vs overridable at `predict()` time.
4. **Bind & Load (`configure` $\rightarrow$ `load`)**: Bind `implementation=YourDetectorClass` in [`models.configure()`](api/configure.md) and instantiate via [`models.load()`](api/load.md).

<details open>
<summary><b>Pattern A: Torchvision Detector (Faster R-CNN, RetinaNet, FCOS, SSD)</b> <i>(Click to expand)</i></summary>

```python
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.transforms.functional import pil_to_tensor
from klygo import models, outputs


class TorchvisionFasterRCNN(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Torchvision", **kwargs)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()
        self.names = {i: name for i, name in enumerate(weights.meta["categories"])}

    def forward(self, images, prompt=None):
        post_cfg = dict(self.metadata.get("post", {}))
        threshold = float(post_cfg.get("threshold", 0.25))

        tensors = [pil_to_tensor(img.convert("RGB")).float() / 255.0 for img in images]
        inputs = self.cast_inputs(tensors)
        raw_outputs = self.run_inference(inputs)
        formatted = self.format_results(raw_outputs)

        for item in formatted:
            keep = [i for i, s in enumerate(item["scores"]) if s >= threshold]
            item["boxes"] = [item["boxes"][i] for i in keep]
            item["scores"] = [item["scores"][i] for i in keep]
            item["class_ids"] = [item["class_ids"][i] for i in keep]
            item["labels"] = [self.names.get(cid, str(cid)) for cid in item["class_ids"]]

        return outputs.build_detections(
            images,
            formatted,
            metadata={"prompt": prompt, **post_cfg},
            names=self.names,
        )
```

</details>

<details open>
<summary><b>Pattern B: Keras 3 / KerasHub Detector (RT-DETR / D-FINE, RetinaNet)</b> <i>(Click to expand)</i></summary>

```python
import numpy as np
import keras_hub
from klygo import models, outputs


class KerasHubDFineDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Keras", **kwargs)
        self.model = keras_hub.models.ObjectDetector.from_preset(self.model_id)
        self.names = {0: "person", 1: "bicycle", 2: "car", 7: "truck"}  # COCO mapping

    def forward(self, images, prompt=None):
        post_cfg = dict(self.metadata.get("post", {}))
        threshold = float(post_cfg.get("threshold", 0.25))

        batch = np.stack([np.asarray(img.convert("RGB"), dtype=np.float32) for img in images], axis=0)
        # Keras backend automatically resizes to converter target, runs decode_predictions,
        # filters -1 padding, and converts yxyx -> xyxy boxes.
        inputs = self.cast_inputs(batch)
        raw_outputs = self.run_inference(inputs)
        formatted = self.format_results(raw_outputs)

        for item in formatted:
            keep = [i for i, s in enumerate(item["scores"]) if s >= threshold]
            item["boxes"] = [item["boxes"][i] for i in keep]
            item["scores"] = [item["scores"][i] for i in keep]
            item["class_ids"] = [item["class_ids"][i] for i in keep]
            item["labels"] = [self.names.get(cid, str(cid)) for cid in item["class_ids"]]

        return outputs.build_detections(
            images,
            formatted,
            metadata={"prompt": prompt, **post_cfg},
            names=self.names,
        )
```

</details>

<details open>
<summary><b>Pattern C: Ultralytics Detector (YOLOv8/v11 &amp; YOLO-World)</b> <i>(Click to expand)</i></summary>

```python
from ultralytics import YOLO
from klygo import models, outputs


class CustomUltralyticsDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Ultralytics", **kwargs)
        self.model = YOLO(self.model_id)
        self.names = dict(self.model.names)

    def forward(self, images, prompt=None):
        post_cfg = dict(self.metadata.get("post", {}))
        results = self.model(
            images,
            conf=post_cfg.get("threshold", 0.25),
            iou=post_cfg.get("iou", 0.7),
            verbose=False,
        )
        return outputs.build_detections(
            images,
            self.format_results(results),
            metadata={"prompt": prompt, **post_cfg},
            names=self.names,
        )
```

</details>

<details open>
<summary><b>Pattern D: Hugging Face Transformers Detector (Grounding DINO / DETR)</b> <i>(Click to expand)</i></summary>

```python
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from klygo import models, outputs


class CustomHFZeroShotDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Hugging Face", **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).eval()

    def forward(self, images, prompt=None):
        post_cfg = dict(self.metadata.get("post", {}))
        labels = models.utils.normalize_prompt(prompt or ["object"])
        inputs = self.processor(images=images, text=[labels] * len(images), return_tensors="pt")
        inputs = self.cast_inputs(inputs)
        raw_outputs = self.run_inference(inputs)

        out_dev = self.get_output_device(raw_outputs)
        input_ids = self.sync_device(inputs.get("input_ids"), out_dev)
        target_sizes = [img.size[::-1] for img in images]
        results = self.processor.post_process_grounded_object_detection(
            raw_outputs,
            input_ids,
            threshold=post_cfg.get("threshold", 0.25),
            text_threshold=post_cfg.get("text_threshold", 0.3),
            target_sizes=target_sizes,
        )
        return outputs.build_detections(images, results, metadata={"prompt": labels, **post_cfg})
```

</details>

<details open>
<summary><b>Step 3 &amp; 4: Configure Schema and Load Any Custom Detector</b> <i>(Click to expand)</i></summary>

```python
from klygo import models, outputs


class MinimalDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Torchvision", **kwargs)
        self.names = {1: "person", 2: "bicycle", 3: "car"}

    def forward(self, images, prompt=None):
        threshold = float(self.metadata.get("post", {}).get("threshold", 0.25))
        raw_predictions = [
            {"boxes": [[12.0, 18.0, 140.0, 220.0]], "scores": [0.94], "labels": ["person"], "class_ids": [1]}
            for _ in images
        ]
        return outputs.build_detections(
            images,
            raw_predictions,
            metadata={"threshold": threshold, "prompt": prompt},
            names=self.names,
        )


flags = models.flags(model=False, post=True)
schema = models.metadata(
    flags,
    priority=models.priority(flags, post=("threshold", "iou")),
    details=models.details(
        name="MinimalDetector",
        task="Object-Detection",
        backend="Torchvision",
        library="torchvision",
    ),
)
configured = models.configure(
    "custom-fasterrcnn",
    metadata=schema,
    implementation=MinimalDetector,
    threshold=0.4,
)
detector = models.load(configured)
```

</details>

For a complete, real-image benchmark running **Ultralytics YOLOv8**, **Ultralytics YOLO-World**, **Torchvision Faster R-CNN**, and **KerasHub RT-DETR (D-FINE)** side by side on COCO15, see [`examples/integration/multi_backend_workflow.py`](../../examples/integration/multi_backend_workflow.py) and the [integration workflows guide](../integration/README.md).
