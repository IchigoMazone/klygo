# Klygo

<p align="center">
  <strong>A unified, high-performance Python toolkit for file operations, archives, configuration, dataset management, and zero-shot object detection pipelines.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-2.1.0-green.svg" alt="Version 2.1.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/files/Document_Files_Overview.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://github.com/IchigoMazone/klygo/actions"><img src="https://img.shields.io/badge/tests-passing-brightgreen.svg" alt="Tests"></a>
</p>

---

## Highlights

- **Zero-RAM & Streaming**: Stream large videos and high-resolution image folders via `media.load(..., stream=True)` with constant memory consumption (< 10 MB RAM).
- **Lazy Transformations with Trace**: Preprocess images lazily with `klygo.preprocessing` while preserving spatial coordinate transforms for automated bounding box restoration (`detection.restore()`).
- **Unified Multi-Backend AI Detection**: Run Hugging Face Transformers, Ultralytics YOLO, Torchvision, and KerasHub through a single, consistent API (`klygo.models.load()`).
- **Standardized Detection Outputs**: Manipulate detections seamlessly using `Box`, `Crops`, `Detection`, and `Detections` collections with built-in cropping, filtering, formatting, and visualization.
- **Production-Grade Data & Archive Operations**: Partition, unpartition, repartition, split, and merge YOLO datasets; manipulate ZIP, TAR, GZip, 7z, and RAR archives cleanly.
- **Interactive Google Colab Tutorials**: Over 60 fully tested tutorial notebooks with automated GitHub-to-Colab synchronization.

---

## Architecture

```text
Source Data (Files / Images / Videos / Archives)
      │
      ▼
klygo.files / klygo.archive / klygo.media (Zero-RAM Streaming)
      │
      ▼
klygo.preprocessing (Lazy transforms with coordinate tracking)
      │
      ▼
klygo.models (Unified Interface: Transformers, YOLO, Torchvision, KerasHub)
      │
      ▼
klygo.outputs (Standardized Box, Crops, Detection, Detections)
      │
      ▼
klygo.postprocessing (Composable pipelines: NMS, Clip, Threshold, Filter)
      │
      ▼
klygo.visual / klygo.datasets (Visualization, Analysis & Dataset Export)
```

---

## Installation

### From Source or Git

```bash
# Clone the repository
git clone https://github.com/IchigoMazone/klygo.git
cd klygo

# Install in editable mode
pip install -e .

# Or using uv
uv sync
```

### Install with Optional Backends

Klygo allows modular installations so you only download the dependencies you need:

| Backend Group | Command | Description |
| :--- | :--- | :--- |
| **Archive: 7z** | `pip install "klygo[py7zr]"` | Read and write `.7z` archives |
| **Archive: RAR** | `pip install "klygo[rarfile]"` | Extract `.rar` archives |
| **Model: Transformers** | `pip install "klygo[transformers]"` | Hugging Face models (Grounding DINO, etc.) |
| **Model: Ultralytics** | `pip install "klygo[ultralytics]"` | Ultralytics YOLO models (YOLOv8, YOLO11, World) |
| **Model: Torchvision** | `pip install "klygo[torchvision]"` | Standard Torchvision detectors (Faster R-CNN, etc.) |
| **Model: KerasHub** | `pip install "klygo[keras-hub]"` | Keras 3 & KerasHub object detection backends |
| **All Models** | `pip install "klygo[models]"` | All supported detection model frameworks |

---

## Module Overview

| Module | Public APIs | Description | Docs | Colab Guide |
| :--- | :---: | :--- | :---: | :---: |
| [`klygo.files`](klygo/files/) | **43** | Path manipulation, filesystem operations, structured data I/O, checksums, and streaming downloads. | [README](docs/files/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/files/Document_Files_Overview.ipynb) |
| [`klygo.archive`](klygo/archive/) | **22** | Create, inspect, extract, search, verify, merge, and split ZIP, TAR, GZip, 7z, and RAR archives. | [README](docs/archive/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/archive/Document_Archive_Overview.ipynb) |
| [`klygo.config`](klygo/config/) | **21** | Multi-format configuration (YAML, JSON, TOML, ENV), dot-path notation, validation, and object access. | [README](docs/config/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/config/Document_Config_Overview.ipynb) |
| [`klygo.datasets`](klygo/datasets/) | **8** | Load, inspect, partition, repartition, unpartition, merge, split, and remap YOLO datasets. | [README](docs/datasets/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/datasets/Document_Datasets_Overview.ipynb) |
| [`klygo.media`](klygo/media/) | **15** | Lazy image and video reading, Zero-RAM streaming, format conversions, probe, and PIL/Tensor export. | [README](docs/media/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/media/Document_Media_Overview.ipynb) |
| [`klygo.models`](klygo/models/) | **15** | Unified AI model registry, auto-download, backend routing, benchmark, and inference execution. | [README](docs/models/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/models/Document_Models_Overview.ipynb) |
| [`klygo.outputs`](klygo/outputs/) | **1** | Standard detection output collections (`Box`, `Crops`, `Detection`, `Detections`). | [README](docs/outputs/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/outputs/Document_Outputs_Overview.ipynb) |
| [`klygo.postprocessing`](klygo/postprocessing/) | **20** | Composable detection pipelines: Non-Maximum Suppression (NMS), confidence thresholding, and box clipping. | [README](docs/postprocessing/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/postprocessing/Document_Postprocessing_Overview.ipynb) |
| [`klygo.preprocessing`](klygo/preprocessing/) | **137** | Lazy image transformations, normalization, color grading, geometric warping, and coordinate tracking. | [README](docs/preprocessing/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/preprocessing/Document_Preprocessing_Overview.ipynb) |
| [`klygo.utils`](klygo/utils/) | **3** | Formatting helpers (`human_size`), string conversions, and interactive progress bars (`ProgressBar`). | [README](docs/utils/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/utils/Document_Utils_Overview.ipynb) |
| [`klygo.validators`](klygo/validators/) | **1** | Strict runtime type and value validation (`validate_type`). | [README](docs/validators/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/validators/Document_Validators_Overview.ipynb) |
| [`klygo.visual`](klygo/visual/) | **3** | Render bounding boxes (`draw_bboxes`), preview images (`show_image`), and plot dataset class distributions. | [README](docs/visual/README.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/visual/Document_Visual_Overview.ipynb) |

---

## Quick Start

### 1. Files & Archives

```python
from klygo import archive, files

# Path construction & verification
work_dir = files.path("workspace")
files.mkdir(work_dir)

# Structured data I/O (auto-detects JSON, YAML, TOML, ENV)
files.save(work_dir / "meta.json", {"project": "klygo", "version": "2.1.0"}, overwrite=True)
data = files.load(work_dir / "meta.json")

# Create & extract compressed bundles
archive.compress(work_dir, "bundle.zip", overwrite=True)
archive.extract("bundle.zip", output_dir="extracted", overwrite=True)
```

### 2. Multi-Format Configuration

```python
from klygo import config
from klygo.config import Config

# Functional dot-path access
cfg = config.load("app.yaml")
config.set(cfg, "server.port", 8080)
port = config.get(cfg, "server.port")

# Object-oriented Config manager
mgr = Config("app.yaml")
mgr.set("database.host", "localhost")
mgr.save(overwrite=True)
```

### 3. Zero-RAM Video & Media Streaming

```python
from klygo import media

# Read video as a lazy stream without loading all frames into RAM
video_stream = media.load("traffic.mp4", stream=True)
print(f"Total frames: {video_stream.total_frames}, FPS: {video_stream.fps}")

for frame in video_stream:
    # frame is loaded into memory only on demand
    print(f"Frame #{frame.frame_index}: size={frame.size}")
```

### 4. Zero-Shot Detection & Coordinate Restoration

```python
from klygo import media, models, postprocessing, preprocessing, visual

# 1. Load an open-vocabulary detector
detector = models.load("grounding-dino-tiny")

# 2. Apply lazy image transformations (records spatial trace)
raw_image = media.load("street.jpg")[0]
transform = preprocessing.compose(
    preprocessing.resize((640, 640)),
    preprocessing.adjust_brightness(1.2),
)
processed_image = transform(raw_image)

# 3. Predict with text prompts
results = detector.predict(processed_image, prompt="car. pedestrian. bicycle.")

# 4. Clean up results with composable postprocessing
cleaned = postprocessing.compose(
    postprocessing.threshold(0.35),
    postprocessing.nms(iou=0.5),
)(results)

# 5. Restore bounding boxes back to the original image coordinates
original_coords = cleaned[0].restore(target="original")
original_coords.show()
```

### 5. Dataset Partitioning & Management

```python
from klygo import datasets

# Inspect dataset information
stats = datasets.info("raw_dataset.zip")
print(f"Classes: {stats['classes']}, Images: {stats['total_images']}")

# Partition into train/val/test splits (80% / 10% / 10%)
partitioned = datasets.partition(
    source="raw_dataset.zip",
    target="partitioned_dataset",
    ratios=(0.8, 0.1, 0.1),
    overwrite=True,
)

# Flatten splits back into a unified dataset collection
flat = datasets.unpartition(partitioned, "flat_dataset")
detections = datasets.load(flat)
print(f"Loaded {len(detections)} annotated detection samples.")
```

---

## Interactive Tutorials

Interactive tutorial notebooks are available in the [`tutorials/`](tutorials/) directory. You can run them locally or execute them instantly in Google Colab:

- **Files**: [Overview](tutorials/files/Document_Files_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/files/Document_Files_Overview.ipynb)
- **Archive**: [Overview](tutorials/archive/Document_Archive_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/archive/Document_Archive_Overview.ipynb)
- **Config**: [Overview](tutorials/config/Document_Config_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/config/Document_Config_Overview.ipynb)
- **Datasets**: [Overview](tutorials/datasets/Document_Datasets_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/datasets/Document_Datasets_Overview.ipynb)
- **Media**: [Overview](tutorials/media/Document_Media_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/media/Document_Media_Overview.ipynb)
- **Models**: [Overview](tutorials/models/Document_Models_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/models/Document_Models_Overview.ipynb)
- **Outputs**: [Overview](tutorials/outputs/Document_Outputs_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/outputs/Document_Outputs_Overview.ipynb)
- **Postprocessing**: [Overview](tutorials/postprocessing/Document_Postprocessing_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/postprocessing/Document_Postprocessing_Overview.ipynb)
- **Preprocessing**: [Overview](tutorials/preprocessing/Document_Preprocessing_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/preprocessing/Document_Preprocessing_Overview.ipynb)
- **Utils**: [Overview](tutorials/utils/Document_Utils_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/utils/Document_Utils_Overview.ipynb)
- **Validators**: [Overview](tutorials/validators/Document_Validators_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/validators/Document_Validators_Overview.ipynb)
- **Visual**: [Overview](tutorials/visual/Document_Visual_Overview.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/visual/Document_Visual_Overview.ipynb)

---

## Quality Assurance & Verification

Klygo enforces strict automated verification across source code, docstrings, examples, and tutorial notebooks:

```bash
# 1. Run full unittest suite
python -m unittest discover -s test -p "test*.py"

# 2. Run repository-wide public API & documentation audit
python .agents/skills/klygo-module-guides/scripts/audit_public_api.py --all

# 3. Verify and synchronize Google Colab badges across all notebooks
python .agents/skills/klygo-module-guides/scripts/audit_public_api.py --sync-colab
```

---

## Authors & License

- Authors & Maintainers: [AUTHORS.md](AUTHORS.md)
- License: [MIT License](LICENSE) © 2026 Trinh Nhu Nhat (IchigoMazone).
