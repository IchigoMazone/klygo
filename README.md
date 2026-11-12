# Klygo

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IchigoMazone/klygo/blob/main/LICENSE)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-blue.svg)](https://deepwiki.com/IchigoMazone/klygo)
[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/IchigoMazone/badge.svg)](https://hub.xerj.org/contributors/IchigoMazone)

Klygo is a modular, composable Python toolkit for files, archives,
configuration, media, dataset management, dataset mining, preprocessing, and
multi-backend object-detection pipelines. Every public module exposes small,
independent APIs backed by shared result types so teams can adopt a single layer
or chain all 13 modules end to end.

Klygo supports **Python 3.10+** and keeps core installation lightweight—heavy
deep-learning frameworks (`torch`, `transformers`, `ultralytics`, `torchvision`,
`keras-hub`) are strictly optional extras.

## Architecture

- `files`, `archive`, `config`, and `media` handle storage, I/O, compression,
  hierarchical settings, and lazy image/video frame streams.
- `datasets` converts, inspects, partitions, merges, splits, and remaps
  YOLO-style datasets.
- `mining` audits dataset statistics, finds exact and perceptual duplicates,
  scores sample quality/rarity, clusters visually, and balances subsets.
- `preprocessing` applies lazy image transforms while tracking coordinate
  history (`trace`) for automatic bounding-box restoration.
- `models` supplies a backend-neutral `Detector` lifecycle and lazy integrations
  for Ultralytics, Hugging Face Transformers, Torchvision, and Keras 3 / KerasHub.
- `outputs`, `postprocessing`, and `visual` standardize detections (`Box`,
  `Detection`, `Detections`), compose NMS/filtering pipelines, and render clean
  annotations and dataset charts.

```text
files / archives / images / videos
                 │
                 ▼
       media + preprocessing (trace)
                 │
                 ▼
      Detector & optional backend
                 │
                 ▼
      normalized prediction mappings
                 │
                 ▼
        outputs.build_detections
                 │
                 ▼
    postprocessing ──► visual / mining
```

A concrete detector owns its framework model through `detector.model`. Calling
`detector(source, ...)` runs the full Klygo lifecycle (lazy loading, trace
restoration, timing, streaming), while `detector.model(...)` accesses the native
framework API directly.

## Installation

Install the lightweight core package from PyPI:

```bash
pip install klygo
```

Or with `uv`:

```bash
uv add klygo
```

For local development from source:

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
pip install -e .
```

### Optional integrations

Install only the optional backends required by your workflow:

| Integration | pip | uv |
| --- | --- | --- |
| Hugging Face Transformers | `pip install "klygo[transformers]"` | `uv add "klygo[transformers]"` |
| Ultralytics | `pip install "klygo[ultralytics]"` | `uv add "klygo[ultralytics]"` |
| Torchvision | `pip install "klygo[torchvision]"` | `uv add "klygo[torchvision]"` |
| KerasHub | `pip install "klygo[keras-hub]"` | `uv add "klygo[keras-hub]"` |
| All model integrations | `pip install "klygo[models]"` | `uv add "klygo[models]"` |
| PyTorch runtime only | `pip install "klygo[torch]"` | `uv add "klygo[torch]"` |
| CuPy GPU outputs (`CUDA 12`) | `pip install "klygo[gpu-outputs]"` | `uv add "klygo[gpu-outputs]"` |
| 7z archives | `pip install "klygo[py7zr]"` | `uv add "klygo[py7zr]"` |
| RAR archives | `pip install "klygo[rarfile]"` | `uv add "klygo[rarfile]"` |

Inside an existing `uv` environment, the equivalent direct installer commands
are:

```bash
uv pip install "klygo[transformers]"
uv pip install "klygo[ultralytics]"
uv pip install "klygo[torchvision]"
uv pip install "klygo[keras-hub]"
uv pip install "klygo[models]"
```

KerasHub can run through TensorFlow, JAX, or PyTorch by selecting
`KERAS_BACKEND`. Install `klygo[keras-hub,torch]` when using its PyTorch
backend.

## Quick start

### Files, configuration, and archives

```python
from klygo import archive, config, files

workspace = files.path("workspace")
files.mkdir(workspace)

settings_path = files.join(workspace, "settings.yaml")
files.save(settings_path, {"model": {"threshold": 0.4}}, overwrite=True)
settings = config.load(settings_path)
settings = config.set(settings, "model.batch", 4)

archive.compress(workspace, "workspace.zip", overwrite=True)
archive.extract("workspace.zip", output_dir="restored", overwrite=True)
```

### Detection and postprocessing

```python
from klygo import models, postprocessing

detector = models.load("grounding-dino-tiny")
results = detector("street.jpg", prompt=["car", "person"])

cleaned = postprocessing.compose(
    postprocessing.threshold(0.4),
    postprocessing.clip(),
    postprocessing.nms(iou=0.5),
)(results)

cleaned[0].show()
```

`prompt` is normalized before `forward()` and shares one query across every
image in a batch. Ordinary YOLO models map prompt names to checkpoint class IDs
for filtering. YOLO-World models use the prompt as their active vocabulary and
retain `original_names` as a snapshot of the vocabulary loaded from the
checkpoint. Advanced Ultralytics operations remain available through
`detector.model`.

### Build output objects from a custom backend

```python
from PIL import Image

from klygo import outputs

image = Image.new("RGB", (640, 480))
frames = outputs.build_detections(
    [image],
    [{
        "boxes": [[40, 50, 240, 300]],
        "scores": [0.93],
        "labels": ["person"],
        "class_ids": [0],
    }],
    metadata={"backend": "custom"},
    names={0: "person"},
)

print(frames[0].labels)
print(frames[0].names)
```

`Box.uid` identifies a box inside its frame, while `Box.class_id` indexes
`names`. `results.names` contains the complete result vocabulary;
`results[index].names` contains only classes present in that frame.

`outputs.build_detections()` expects one normalized prediction mapping per
image and returns `list[Detection]`. `Detector.predict()` adds timing, source,
frame, and streaming information before returning `Detections`.

### Datasets and data mining

```python
from klygo import datasets, mining

samples = datasets.load("flat-dataset", format="yolo", verbose=False)
summary = mining.stats(samples)
print("Imbalance ratio:", summary.imbalance_ratio)

dedup = mining.deduplicate(samples, keep="best", threshold=0.92)
balanced = mining.balance(dedup, target="median", count_by="images")
print(mining.compare(samples, balanced).changes)
```

Dataset transformations expect a YOLO-style dataset root or a supported
archive containing one. On a noisy long-tail 10-class COCO `val2017` pool
(`848` train images, `29.1x` class imbalance, evaluated on `247` clean held-out
images on NVIDIA L4), `mining.deduplicate()` + `mining.balance()` improved
downstream `YOLOv8n` `mAP@50-95` from `0.3139` to `0.3874` (`+23.4%` relative)
using `6.0x` fewer training images (`142` vs `848`) and `2.4x` less training time.

### Remove consumed options

```python
from klygo import utils

options = {"threshold": 0.4, "dtype": "float16", "device": "cuda"}
model_options = utils.exclude_keys(options, ["dtype", "device"])
```

`exclude_keys()` returns a shallow copy and never mutates the source mapping.

## Modules

| Module | Responsibility | Documentation |
| --- | --- | --- |
| `klygo.files` | Paths, filesystem operations, downloads, and structured-data I/O | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/files/README.md) |
| `klygo.archive` | Create, inspect, extract, convert, split, and merge archives | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/archive/README.md) |
| `klygo.config` | Hierarchical configuration and model metadata | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/config/README.md) |
| `klygo.media` | Lazy image and video loading, conversion, streaming, and saving | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/media/README.md) |
| `klygo.preprocessing` | Lazy image transforms, metrics, and coordinate restoration | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/preprocessing/README.md) |
| `klygo.models` | Detector lifecycle, metadata, loading, and backend hooks | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/models/README.md) |
| `klygo.outputs` | Structured detection results and normalized-result construction | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/outputs/README.md) |
| `klygo.postprocessing` | Backend-independent detection transformations | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/postprocessing/README.md) |
| `klygo.datasets` | Convert, load, inspect, partition, merge, split, and relabel datasets | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/datasets/README.md) |
| `klygo.mining` | Analyze, deduplicate, cluster, select, and balance detection datasets | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/mining/README.md) |
| `klygo.visual` | Display images, boxes, and dataset statistics | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/visual/README.md) |
| `klygo.utils` | Shared formatting, mapping, and progress helpers | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/utils/README.md) |
| `klygo.validators` | Reusable runtime validation | [Guide](https://github.com/IchigoMazone/klygo/blob/main/docs/validators/README.md) |

See the full [documentation index](https://github.com/IchigoMazone/klygo/blob/main/docs/README.md),
[cross-module integration guides](https://github.com/IchigoMazone/klygo/blob/main/docs/integration/README.md),
and [benchmark reports](https://github.com/IchigoMazone/klygo/blob/main/benchmarks/README.md).

## Development

Run the standard-library test suite:

```bash
python -m unittest discover -s test -t . -p "test*.py"
```

Audit public APIs, examples, documentation, and tutorials:

```bash
python .agents/skills/klygo-module-guides/scripts/audit_public_api.py --all
```

Runtime datasets, downloaded weights, generated reports, and other artifacts
belong outside source, test, example, documentation, and tutorial directories.
Repository ignore rules keep those generated assets out of version control.

## License

Released under the [MIT License](https://github.com/IchigoMazone/klygo/blob/main/LICENSE).
