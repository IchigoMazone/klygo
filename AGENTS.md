# AGENTS.md — AI Coding Assistant Guide for Klygo

This document provides authoritative architectural context, API conventions, dogfooding rules, and verification commands for AI coding agents working in or building applications with the **Klygo** repository.

---

## 1. Architecture & 13 Public Modules

Klygo is a modular, composable Python toolkit (`Python >= 3.10`) organized into four cooperative layers across **13 public modules**:

```text
1. Storage, Config & Datasets : klygo.files • klygo.archive • klygo.config • klygo.datasets • klygo.mining
2. Media & Preprocessing      : klygo.media (LazyImage, MediaFrames) • klygo.preprocessing (trace)
3. Multi-Backend AI Models    : klygo.models (BaseModel -> Detector -> native model wrappers)
                                ├── Ultralytics (YOLOv8/v11, YOLO-World v2)
                                ├── Hugging Face / PyTorch (Grounding DINO, DETR, custom nn.Module)
                                ├── Torchvision (Faster R-CNN, RetinaNet, FCOS, SSD)
                                └── Keras 3 / KerasHub (RT-DETR / D-FINE, RetinaNet on torch/jax/tf)
4. Outputs, Post, Reports & Graphics : klygo.outputs • klygo.postprocessing • klygo.metrics • klygo.reports • klygo.graphics
   Shared Foundation          : klygo.utils (ProgressBar, human_size) • klygo.validators
```

| Module | Package Import | Primary Responsibility | Reference Docs |
| :--- | :--- | :--- | :--- |
| `files` | `from klygo import files` | 43 path, filesystem, checksum, download, and 14-format structured data I/O helpers | [`docs/files/README.md`](docs/files/README.md) |
| `archive` | `from klygo import archive` | Create, inspect, extract, search, convert, merge, and split ZIP, TAR, GZip, 7z, and RAR archives | [`docs/archive/README.md`](docs/archive/README.md) |
| `config` | `from klygo import config, Config` | Immutable functional config helpers (`load`, `get`, `set`, `merge`, `diff`, `from_env`) & stateful `Config` | [`docs/config/README.md`](docs/config/README.md) |
| `datasets` | `from klygo import datasets` | Convert, inspect, load, partition, repartition, unpartition, merge, split, and remap YOLO datasets | [`docs/datasets/README.md`](docs/datasets/README.md) |
| `mining` | `from klygo import mining` | Analyze, deduplicate, cluster, select, balance, and compare object-detection datasets at image level | [`docs/mining/README.md`](docs/mining/README.md) |
| `media` | `from klygo import media` | Lazy image (`LazyImage`) and unified indexed/streaming frames (`MediaFrames`), frame iteration, and PIL/NumPy/Tensor conversion | [`docs/media/README.md`](docs/media/README.md) |
| `preprocessing` | `from klygo import preprocessing as p` | 137 lazy transforms, image quality metrics/predicates, and automatic coordinate restoration (`trace`) | [`docs/preprocessing/README.md`](docs/preprocessing/README.md) |
| `models` | `from klygo import models` | 3-tier model wrappers (`BaseModel`, `Detector`), schema builders (`metadata`, `flags`, `priority`, `details`, `configure`), and `load` | [`docs/models/README.md`](docs/models/README.md) |
| `outputs` | `from klygo import outputs` | Standardized detection containers (`Box`, `Crops`, `Detection`, `Detections`) and `build_detections` | [`docs/outputs/README.md`](docs/outputs/README.md) |
| `postprocessing` | `from klygo import postprocessing as post` | Composable and direct box operations (`nms`/`suppress`, `threshold`/`filter_score`, `clip`/`clip_boxes`, `box_iou`, `top`) | [`docs/postprocessing/README.md`](docs/postprocessing/README.md) |
| `reports` | `from klygo import reports` | Experiment log loading (`load`), `partition_summary`, multi-seed `aggregate_runs`, `threshold_audit`, and `mining_breakdown` | [`docs/reports/README.md`](docs/reports/README.md) |
| `graphics` | `from klygo import graphics` | Non-destructive rendering (`draw`, `show`, `compare`, `save`), grids (`cell`, `compose`, `grid`), and charts (`dataset_stats`, `partitions`, `curves`, `thresholds`, `mining`, `bars`) | [`docs/graphics/README.md`](docs/graphics/README.md) |
| `utils` | `from klygo import utils` | Mapping helpers (`exclude_keys`, `invert`), binary byte formatting (`human_size`), and `ProgressBar` | [`docs/utils/README.md`](docs/utils/README.md) |
| `validators` | `from klygo import validators` | Runtime type validation (`validate_type`) raising standardized `TypeError` messages | [`docs/validators/README.md`](docs/validators/README.md) |

---

## 2. Golden Rules for AI Agents

### Rule 0: High-Level First — Use the Owning Module's Public API

This is the default for proposed code, answers, application workflows, examples,
documentation, notebooks, and their generators. Apply it before cross-module
dogfooding. See [API levels and selection](docs/API_LEVELS.md).

- **High-level API** means the supported public surface of a module: its exported
  functions/classes and documented public object methods. `files.copy()` is a
  high-level API of `files`, just as `datasets.partition()` is one of `datasets`.
- **Low-level implementation** means assembling underlying operations to recreate
  a capability already exposed by the module responsible for that capability.
  It can use only public Klygo calls and still be the wrong default: manually
  splitting image/label files with `files` recreates `datasets.partition()`.
- A high-level API may internally compose other modules' high-level APIs. Reuse
  that surface from application code instead of copying its implementation.
  Internal/private APIs are a separate concept, not a synonym for low-level code.

Before proposing a helper, wrapper, subclass, loop, parser, converter, metric, or
export routine: identify the user's goal and owning module, inspect the current
`__all__`, public object methods, implementation/signature, and relevant example.
Then follow this order:

1. If an existing public API covers the requested semantics, call it directly.
2. If several steps are needed, compose the existing public APIs. Do not recreate
   the steps they already perform, or maximize the number of APIs called.
3. If a verified gap remains, implement only that gap and retain existing APIs for
   everything else. Briefly explain the missing capability and custom boundary.
4. Use a fully custom implementation only for an explicit custom/low-level request
   or when no suitable public API exists. Do not invent an API or assume a gap
   from an unfamiliar name or an old example; read the current implementation.

Examples: use `models.load()` and the detector's public `predict()` instead of
rebuilding supported inference; `mining.deduplicate()` instead of manually
hashing/grouping/removing images; `metrics.evaluate()` or the appropriate direct
metric instead of rewriting IoU matching and TP/FP/FN aggregation;
`Detections.export()` instead of manually writing a supported detection dataset.
Do not wrap a single existing API under a new helper name without a real reuse
or contract boundary. Show the direct supported call first in ordinary examples.

Library implementation, unsupported backend adapters, focused internal tests,
and explicitly requested raw-reference benchmarks may use underlying operations
where necessary for their purpose. Keep those operations inside that boundary.
An API being implemented is not an alternative to its own implementation; avoid
recursive delegation. These exceptions do not justify rebuilding an existing
workflow in a user-facing example.

### Rule 1: Always Import Public Package Facades
Import modules from top-level `klygo`:
```python
from klygo import archive, config, datasets, files, graphics, media, metrics, mining, models, outputs, postprocessing, preprocessing, reports, utils, validators
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
1. **Layer 1 (`models.BaseModel` in `klygo/models/base.py`)**: Manages `metadata`, `details`, `flags`, `state` (`LOADING`, `READY`), and `info()`.
2. **Layer 2 (`models.Detector` in `klygo/models/detection/base.py`)**: Implements `predict(source, prompt=None, stream=False, **kwargs)`, automatic coordinate restoration from `klygo.preprocessing` traces, `warmup()`, `benchmark()`, with native wrapper inference and shared accelerator lifecycle helpers in `klygo.models.runtime`.
3. **Layer 3 (Concrete Wrappers & Custom Subclasses)**: Implement `forward(self, images: list[PIL.Image.Image], prompt=None) -> list[Detection]`.

### 3.2. Custom Adapters Only When Needed

For ordinary inference, inspect `models.load()` and the existing adapters first,
then call the loaded detector's `predict()`. Do not subclass a detector merely to
wrap an already supported architecture or to demonstrate module integration.
Use the pattern below only for an explicit custom-model request or a verified
missing adapter, keeping preprocessing, evaluation and export on existing public
APIs outside the unsupported backend boundary.

**Mandatory custom-detector contract:** When implementing a custom object detector
inside Klygo, the class **must inherit `models.Detector`** (directly or through an
existing Detector subclass) and **must implement `forward(images, prompt=None, ...)`**
for the custom backend behavior. Inheriting only `BaseModel` is not sufficient for
a custom detector. Call the inherited detector `predict()` or `Detections.repredict(detector, ...)`
from workflow code; do not replace the framework's inference orchestration with a
standalone custom `predict()` loop. This requirement applies when customization
is needed, not to ordinary use of a supported detector.

### Standard 4-Step Pattern for Subclassing `models.Detector`

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

        # 3. Optionally use Detector helpers, or own native tensor/model calls
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
- **Ultralytics (`backend="Ultralytics"`)**: Built-in `YOLODetect` (`yolov8n.pt`) and `YOLOWorldDetect` (`yolov8s-worldv2.pt`). Passing a `.pt` path directly to `models.load("yolov8n.pt")` auto-instantiates `YOLODetect`. See [`docs/models/execution/api/ultralytics.md`](docs/models/execution/api/ultralytics.md).
- **Hugging Face (`backend="Hugging Face"`)**: Built-in `GroundingDinoDetect`. `self.cast_inputs(batch_feature)` casts float tensors (`pixel_values`) while keeping `input_ids` as `int64`. Use `out_dev = self.get_output_device(raw_outputs)` and `self.sync_device(inputs["input_ids"], out_dev)` for multi-GPU safety. See [`docs/models/execution/api/huggingface.md`](docs/models/execution/api/huggingface.md).
- **Torchvision (`backend="Torchvision"`)**: Generic adapter for caller-owned models (e.g., `fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)`). Expects `list[Tensor[C, H, W]]` in `[0.0, 1.0]`. See [`docs/models/execution/api/torchvision.md`](docs/models/execution/api/torchvision.md).
- **Keras 3 / KerasHub (`backend="Keras"`)**: Supports `KERAS_BACKEND="torch"` without requiring TensorFlow (`models.set_backend("Keras", engine="torch")`). Natively handles `keras_hub.models.ObjectDetector.from_preset("dfine_small_coco")` (RT-DETR / D-FINE): `run_inference` resizes inputs to `image_converter.image_size` prior to `model.preprocessor` and calls `model.decode_predictions(raw_preds, inputs)`, while `format_results` filters `-1` padding slots and converts `yxyx` $\rightarrow$ `xyxy` boxes automatically. See [`docs/models/execution/api/keras.md`](docs/models/execution/api/keras.md).

---

## 4. End-to-End Workflow Through Public APIs

Compose only the stages the task needs. For a materialized flat YOLO dataset and
a supported detector with matching class semantics, use the existing public
orchestration instead of reconstructing prediction identities or label writers:

```python
from klygo import datasets, metrics, models

truth = datasets.load("artifacts/datasets/flat_yolo", format="yolo", verbose=False)
detector = models.load("yolov8n.pt")

# repredict updates its collection and returns None; preserve ground truth.
predictions = truth.copy()
predictions.repredict(detector, threshold=0.35, verbose=False)

# repredict preserves source identities for the existing metrics API.
report = metrics.evaluate(predictions, truth, verbose=False)
predictions.export("artifacts/integration/predictions", overwrite=True)
predictions.save_images("artifacts/integration/previews", overwrite=True)
```

Use `detector.predict(source, ...)` for ordinary image/video inference.
Use `Detections.repredict()` when replacing predictions in an existing collection
while retaining its source identities. `datasets.load()` returns `Detections`,
not dictionaries with an `image_path` key. Verify input/output contracts rather
than copying an older example. Add preprocessing, postprocessing, selection or
visualization only when the requested task needs those stages; retain their
existing public APIs instead of building custom loops.

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
# 1. Audit all 13 modules (docstrings, API docs, runnable examples, dogfooding, links, and Colab notebooks)
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
