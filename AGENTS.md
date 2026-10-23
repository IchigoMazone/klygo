# AGENTS.md — AI Coding Assistant Guide for Klygo

This document provides authoritative architectural context, API conventions, dogfooding rules, and verification commands for AI coding agents working in or building applications with the **Klygo** repository.

---

## 1. Architecture & 12 Public Modules

Klygo is a modular, composable Python toolkit (`Python >= 3.10`) organized into four cooperative layers across **12 public modules**:

```text
1. Storage, Config & Datasets : klygo.files • klygo.archive • klygo.config • klygo.datasets
2. Media & Preprocessing      : klygo.media (LazyImage, MediaStream) • klygo.preprocessing (trace)
3. Multi-Backend AI Models    : klygo.models (BaseModel -> Detector -> backend.common)
                                ├── Ultralytics (YOLOv8/v11, YOLO-World v2)
                                ├── Hugging Face / PyTorch (Grounding DINO, DETR, custom nn.Module)
                                ├── Torchvision (Faster R-CNN, RetinaNet, FCOS, SSD)
                                └── Keras 3 / KerasHub (RT-DETR / D-FINE, RetinaNet on torch/jax/tf)
4. Outputs, Post & Visual     : klygo.outputs • klygo.postprocessing • klygo.visual
   Shared Foundation          : klygo.utils (ProgressBar, human_size) • klygo.validators
```

| Module | Package Import | Primary Responsibility | Reference Docs |
| :--- | :--- | :--- | :--- |
| `files` | `from klygo import files` | 43 path, filesystem, checksum, download, and 14-format structured data I/O helpers | [`docs/files/README.md`](docs/files/README.md) |
| `archive` | `from klygo import archive` | Create, inspect, extract, search, convert, merge, and split ZIP, TAR, GZip, 7z, and RAR archives | [`docs/archive/README.md`](docs/archive/README.md) |
| `config` | `from klygo import config, Config` | Immutable functional config helpers (`load`, `get`, `set`, `merge`, `diff`, `from_env`) & stateful `Config` | [`docs/config/README.md`](docs/config/README.md) |
| `datasets` | `from klygo import datasets` | Inspect (`info`), load (`load`), partition, repartition, unpartition, merge, split, and remap YOLO datasets | [`docs/datasets/README.md`](docs/datasets/README.md) |
| `media` | `from klygo import media` | Lazy image (`LazyImage`) & video streaming (`MediaStream`), frame iteration, and PIL/NumPy/Tensor conversion | [`docs/media/README.md`](docs/media/README.md) |
| `preprocessing` | `from klygo import preprocessing as p` | 137 lazy transforms, image quality metrics/predicates, and automatic coordinate restoration (`trace`) | [`docs/preprocessing/README.md`](docs/preprocessing/README.md) |
| `models` | `from klygo import models` | 3-tier model wrappers (`BaseModel`, `Detector`), schema builders (`metadata`, `flags`, `priority`, `details`, `configure`), and `load` | [`docs/models/README.md`](docs/models/README.md) |
| `outputs` | `from klygo import outputs` | Standardized detection containers (`Box`, `Crops`, `Detection`, `Detections`) and `build_detections` | [`docs/outputs/README.md`](docs/outputs/README.md) |
| `postprocessing` | `from klygo import postprocessing as post` | Composable and direct box operations (`nms`/`suppress`, `threshold`/`filter_score`, `clip`/`clip_boxes`, `box_iou`, `top`) | [`docs/postprocessing/README.md`](docs/postprocessing/README.md) |
| `visual` | `from klygo import visual` | Non-destructive two-pass box rendering (`draw_bboxes`), adaptive image display (`show_image`), and dataset charts (`plot_dataset_stats`) | [`docs/visual/README.md`](docs/visual/README.md) |
| `utils` | `from klygo import utils` | Mapping helpers (`exclude_keys`, `invert`), binary byte formatting (`human_size`), and `ProgressBar` | [`docs/utils/README.md`](docs/utils/README.md) |
| `validators` | `from klygo import validators` | Runtime type validation (`validate_type`) raising standardized `TypeError` messages | [`docs/validators/README.md`](docs/validators/README.md) |

---

## 2. Golden Rules for AI Agents

### Rule 1: Always Import Public Package Facades
Import modules from top-level `klygo`:
```python
from klygo import archive, config, datasets, files, media, models, outputs, postprocessing, preprocessing, utils, validators, visual
```
Never import private internal helpers (`_helper`) in user-facing examples, tutorials, or workflows.

### Rule 2: Mandatory Cross-Module Dogfooding (No Raw `pathlib` / `os.path` / `shutil` / `json` / `zipfile`)
Inside Klygo modules, examples (`examples/`), tutorials (`tutorials/`), and integration workflows, **always use Klygo's own public APIs** instead of standard-library or third-party equivalents:

| Instead of... | Always use Klygo API |
| :--- | :--- |
| `from pathlib import Path; Path(p)` | `from klygo import files; files.path(p)` ( prohibited in `examples/` by audit script!) |
| `os.path.join(a, b)` | `files.join(a, b)` or `files.path(a) / b` |
| `os.path.exists(p)` / `Path(p).is_file()` / `Path(p).is_dir()` | `files.exists(p)` / `files.is_file(p)` / `files.is_dir(p)` |
| `os.makedirs(p, exist_ok=True)` | `files.mkdir(p)` |
| `shutil.rmtree(p)` / `os.remove(p)` | `files.remove(p, missing_ok=True)` |
| `shutil.copy(src, dst)` / `shutil.move(src, dst)` | `files.copy(src, dst, overwrite=True)` / `files.move(src, dst, overwrite=True)` |
| `json.load` / `yaml.safe_load` / `toml.load` / `Path.read_text` | `files.load(path, verbose=False)` (supports 14 formats automatically) |
| `json.dump` / `yaml.dump` / `Path.write_text` | `files.save(path, data, overwrite=True, verbose=False)` |
| `zipfile.ZipFile` / `tarfile.open` | `archive.compress(...)`, `archive.extract(...)`, `archive.list_files(...)` |
| `{k: v for k, v in d.items() if k not in keys}` | `utils.exclude_keys(d, keys)` |
| `{v: k for k, v in d.items()}` | `utils.invert(d)` |
| `if not isinstance(x, T): raise TypeError(...)` | `validators.validate_type(x, T, "param_name")` |

### Rule 3: Strict English-Only Source & Documentation Policy
- Every `raise <Exception>("...")` error message across the entire codebase must be written in clear, concise, professional English.
- All docstrings (NumPy style), inline comments, markdown files in `docs/`, and notebooks in `tutorials/` must be written strictly in English (enforced automatically by `audit_public_api.py`).

### Rule 4: Pure Documentation Boundary (`docs/`)
- The `docs/` directory must contain **strictly `.md` Markdown files**. Never place `.py` scripts, test files, or `__pycache__` inside `docs/`.
- Every relative link `[text](relative/path.md)` and code fence in `docs/` is verified by `audit_public_api.py`.

### Rule 5: Clean Separation of Code and Artifacts (`artifacts/`)
- Never save generated datasets, downloaded weights (`.pt`, `.keras`, `.safetensors`), preview images, or JSON evaluation reports inside `klygo/`, `examples/`, `test/`, `docs/`, or `tutorials/`.
- Write all runtime outputs under `artifacts/` (e.g., `artifacts/integration/<run-id>/`) or a `tempfile.TemporaryDirectory()`.
- Never commit or push files from `artifacts/`, `weights/`, or `*.pt` checkpoints to Git.

---

## 3. Working with `klygo.models` & Multi-Backend Detection

### 3.1. 3-Tier Model Hierarchy
1. **Layer 1 (`models.BaseModel` in `klygo/models/base.py`)**: Manages `metadata`, `details`, `flags`, `state` (`LOADING`, `READY`, `UNLOADED`), `info()`, and post-unload access guards (calling `predict()` after `unload()` raises `models.errors.InvalidStateError`).
2. **Layer 2 (`models.Detector` in `klygo/models/detection/base.py` & `klygo.models.backend.common`)**: Implements `predict(source, prompt=None, stream=False, **kwargs)`, automatic coordinate restoration from `klygo.preprocessing` traces, `warmup()`, `benchmark()`, `save()`, `reset()`, `unload()`, and delegates tensor/device operations to [`klygo.models.backend.common`](docs/models/backend/api/common.md).
3. **Layer 3 (Concrete Wrappers & Custom Subclasses)**: Implement `forward(self, images: list[PIL.Image.Image], prompt=None) -> list[Detection]`.

### 3.2. Standard 4-Step Pattern for Subclassing `models.Detector`

When integrating any model from **Ultralytics**, **Hugging Face**, **Torchvision**, or **Keras 3 / KerasHub**:

```python
from klygo import models, outputs


class CustomDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        # 1. Initialize Detector with explicit backend ("Ultralytics", "Hugging Face", "Torchvision", or "Keras")
        super().__init__(metadata=metadata, backend="Torchvision", **kwargs)
        # Attach framework model to self.model and class map to self.names
        self.model = ...
        self.names = {0: "person", 1: "car"}

    def forward(self, images, prompt=None):
        # 2. Read active runtime parameters from self.metadata["post"]
        post_cfg = dict(self.metadata.get("post", {}))
        threshold = float(post_cfg.get("threshold", 0.25))

        # 3. Delegate tensor casting -> forward execution -> result formatting to backend helpers
        inputs = self.cast_inputs(preprocessed_tensors)
        raw_outputs = self.run_inference(inputs)
        formatted = self.format_results(raw_outputs)

        # 4. Pack normalized [{'boxes', 'scores', 'labels', 'class_ids'}] into Detection frames
        return outputs.build_detections(
            images,
            formatted,
            metadata={"prompt": prompt, **post_cfg},
            names=self.names,
        )


# Configure metadata schema and load
flags = models.flags(model=False, post=True)
schema = models.metadata(
    flags,
    priority=models.priority(flags, post=("threshold", "iou")),
    details=models.details(
        name="CustomDetector",
        task="Object-Detection",
        backend="Torchvision",
        library="torchvision",
    ),
)
configured = models.configure(
    "my-model-id",
    metadata=schema,
    implementation=CustomDetector,
    threshold=0.4,
)
detector = models.load(configured)
```

### 3.3. Backend-Specific Notes
- **Ultralytics (`backend="Ultralytics"`)**: Built-in `YOLODetect` (`yolov8n.pt`) and `YOLOWorldDetect` (`yolov8s-worldv2.pt`). Passing a `.pt` path directly to `models.load("yolov8n.pt")` auto-instantiates `YOLODetect`. See [`docs/models/backend/api/ultralytics.md`](docs/models/backend/api/ultralytics.md).
- **Hugging Face (`backend="Hugging Face"`)**: Built-in `GroundingDinoDetect`. `self.cast_inputs(batch_feature)` casts float tensors (`pixel_values`) while keeping `input_ids` as `int64`. Use `out_dev = self.get_output_device(raw_outputs)` and `self.sync_device(inputs["input_ids"], out_dev)` for multi-GPU safety. See [`docs/models/backend/api/huggingface.md`](docs/models/backend/api/huggingface.md).
- **Torchvision (`backend="Torchvision"`)**: Generic adapter for caller-owned models (e.g., `fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)`). Expects `list[Tensor[C, H, W]]` in `[0.0, 1.0]`. See [`docs/models/backend/api/torchvision.md`](docs/models/backend/api/torchvision.md).
- **Keras 3 / KerasHub (`backend="Keras"`)**: Supports `KERAS_BACKEND="torch"` without requiring TensorFlow (`models.set_backend("Keras", engine="torch")`). Natively handles `keras_hub.models.ObjectDetector.from_preset("dfine_small_coco")` (RT-DETR / D-FINE): `run_inference` resizes inputs to `image_converter.image_size` prior to `model.preprocessor` and calls `model.decode_predictions(raw_preds, inputs)`, while `format_results` filters `-1` padding slots and converts `yxyx` $\rightarrow$ `xyxy` boxes automatically. See [`docs/models/backend/api/keras.md`](docs/models/backend/api/keras.md).

---

## 4. End-to-End Pipeline Chaining Pattern

When writing an end-to-end Computer Vision workflow in Klygo, chain the public modules in this order:

```python
from klygo import datasets, files, media, models, outputs, postprocessing, preprocessing, visual

# 1. Inspect and load dataset ground truth
info = datasets.info("artifacts/integration/coco15", verbose=False)
samples = datasets.load("artifacts/integration/coco15", format="yolo", verbose=False)

# 2. Load lazy image and apply coordinate-tracked preprocessing
lazy_img = media.load(samples[0]["image_path"], verbose=False)
transform = preprocessing.compose(
    preprocessing.letterbox((640, 640)),
    preprocessing.autocontrast(),
)
processed_img = transform(lazy_img)

# 3. Run detector prediction (automatically restores boxes via preprocessing trace)
detections = detector.predict(processed_img, threshold=0.35)

# 4. Apply backend-neutral postprocessing pipeline (NMS, clipping, top-k)
pipeline = postprocessing.compose(
    postprocessing.clip(),
    postprocessing.nms(iou=0.5),
    postprocessing.top(20),
)
cleaned = pipeline(detections)

# 5. Match against ground truth with postprocessing.box_iou and render visual preview
annotated = visual.draw_bboxes(lazy_img, cleaned[0])
media.save(annotated, "artifacts/integration/preview/result.jpg", overwrite=True, verbose=False)
```

---

## 5. Repository Structure & 1-to-1 Alignment Contract

When adding or modifying a public API in `klygo/<module>/<submodule>.py`, keep all five views synchronized:

1. **Source (`klygo/<module>/`)**:
   - Export symbol in both `klygo/<module>/<submodule>.py` (`__all__`) and `klygo/<module>/__init__.py` (`__all__`).
   - Provide an English NumPy-style docstring.
2. **Unit Tests (`test/<module>/`)**:
   - Standard-library `unittest` suite using `tempfile.TemporaryDirectory()`.
3. **Standalone Example (`examples/<module>/<api_name>.py`)**:
   - One runnable script per exported API in `klygo.<module>.__all__`, using `files.path()` instead of `from pathlib import Path`.
4. **Reference Docs (`docs/<module>/README.md` & `docs/<module>/api/<api_name>.md`)**:
   - Verified signature, contract, parameters, returns, errors, links to example/tests, and complete executable example.
5. **Interactive Tutorials (`tutorials/<module>/`)**:
   - Overview notebook: `tutorials/<module>/Document_<Module>_Overview.ipynb` (with `## Choose a guide` and `## Remember`).
   - 1-to-1 submodule notebooks: `tutorials/<module>/Document_<Module>_<Submodule>.ipynb` (with Colab badge pointing to `v2-release`, `## Jump to an API`, `### <module>.<api>`, `## Put it together`, and `## Remember`).

---

## 6. Verification & Quality Gate Commands

Always run these commands from the repository root (`c:\Users\Mine\Downloads\klygo`) before finishing a task:

```powershell
# 1. Audit all 12 modules (docstrings, API docs, runnable examples, dogfooding, links, and Colab notebooks)
.venv\Scripts\python .agents/skills/klygo-module-guides/scripts/audit_public_api.py --all

# 2. Audit a single module during focused edits
.venv\Scripts\python .agents/skills/klygo-module-guides/scripts/audit_public_api.py models

# 3. Synchronize Colab badges and navigation links across all tutorial notebooks
.venv\Scripts\python .agents/skills/klygo-module-guides/scripts/audit_public_api.py --sync-colab

# 4. Run unit tests
.venv\Scripts\python -m unittest discover -s test -t . -p "test*.py"

# 5. Run the 4-backend real-data evaluation workflow (Ultralytics YOLO, YOLO-World, Torchvision Faster R-CNN, KerasHub D-FINE)
.venv\Scripts\python examples/integration/multi_backend_workflow.py
```
