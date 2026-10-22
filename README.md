# Klygo

Klygo is a composable Python toolkit for files, archives, configuration,
media, datasets, and object-detection workflows. It provides small public APIs
that can be used independently and shared result types that let the modules
work together.

Klygo supports Python 3.10 and newer.

## Design

- `files`, `archive`, `config`, and `media` own storage and input handling.
- `preprocessing` transforms images while retaining coordinate history.
- `models` supplies a backend-neutral `Detector` lifecycle and optional
  integrations for Hugging Face, Ultralytics, Torchvision, and KerasHub.
- `outputs` converts normalized predictions into `Box`, `Detection`, and
  `Detections` objects.
- `postprocessing` transforms those results independently of the model.
- `datasets` loads, inspects, and restructures existing YOLO-style datasets.

```text
files / archives / images / videos
                 │
                 ▼
       media + preprocessing
                 │
                 ▼
      Detector and model backend
                 │
                 ▼
     normalized prediction mappings
                 │
                 ▼
       outputs.build_detections
                 │
                 ▼
          postprocessing
```

A concrete detector owns its framework model through `detector.model`.
Calling `detector(source, ...)` runs the Klygo prediction lifecycle; calling
`detector.model(...)` accesses the framework API directly. Model-specific
`forward()` implementations perform inference and normalize results, while
postprocessing remains a separate workflow step.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
pip install -e .
```

With `uv`:

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
uv sync
```

Install only the optional integrations required by your workflow:

| Integration | pip | uv |
| --- | --- | --- |
| Hugging Face Transformers | `pip install "klygo[transformers]"` | `uv add "klygo[transformers]"` |
| Ultralytics | `pip install "klygo[ultralytics]"` | `uv add "klygo[ultralytics]"` |
| Torchvision | `pip install "klygo[torchvision]"` | `uv add "klygo[torchvision]"` |
| KerasHub | `pip install "klygo[keras-hub]"` | `uv add "klygo[keras-hub]"` |
| All model integrations | `pip install "klygo[models]"` | `uv add "klygo[models]"` |
| PyTorch runtime only | `pip install "klygo[torch]"` | `uv add "klygo[torch]"` |
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

`prompt` is normalized before `forward()` and is shared by every image in a
batch. Ordinary YOLO models map prompt names to checkpoint class IDs for
filtering. YOLO-World models use the prompt as their active vocabulary and retain
`original_names` as a snapshot of the vocabulary loaded from the checkpoint.
Advanced Ultralytics operations remain available through `detector.model`.

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

`Box.id` identifies a box inside its frame, while `Box.class_id` indexes
`names`. `results.names` contains the complete result vocabulary;
`results[index].names` contains only classes present in that frame.

`outputs.build_detections()` expects one normalized prediction mapping per
image and returns `list[Detection]`. `Detector.predict()` adds timing, source,
frame, and streaming information before returning `Detections`.

### Load and restructure a dataset

```python
from klygo import datasets

info = datasets.info("dataset.zip")
flat_path = datasets.unpartition("dataset.zip", "flat-dataset")
samples = datasets.load(flat_path)

print(info["classes"])
print(samples.total_objects)
```

Dataset transformations expect a YOLO-style dataset root or a supported
archive containing one. Dataset generation from model predictions belongs to
the output workflow rather than `klygo.datasets`.

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
| `klygo.files` | Paths, filesystem operations, downloads, and structured-data I/O | [Guide](docs/files/README.md) |
| `klygo.archive` | Create, inspect, extract, convert, split, and merge archives | [Guide](docs/archive/README.md) |
| `klygo.config` | Hierarchical configuration and model metadata | [Guide](docs/config/README.md) |
| `klygo.media` | Lazy image and video loading, conversion, streaming, and saving | [Guide](docs/media/README.md) |
| `klygo.preprocessing` | Lazy image transforms, metrics, and coordinate restoration | [Guide](docs/preprocessing/README.md) |
| `klygo.models` | Detector lifecycle, metadata, loading, and backend hooks | [Guide](docs/models/README.md) |
| `klygo.outputs` | Structured detection results and normalized-result construction | [Guide](docs/outputs/README.md) |
| `klygo.postprocessing` | Backend-independent detection transformations | [Guide](docs/postprocessing/README.md) |
| `klygo.datasets` | Load, inspect, partition, merge, split, and relabel datasets | [Guide](docs/datasets/README.md) |
| `klygo.visual` | Display images, boxes, and dataset statistics | [Guide](docs/visual/README.md) |
| `klygo.utils` | Shared formatting, mapping, and progress helpers | [Guide](docs/utils/README.md) |
| `klygo.validators` | Reusable runtime validation | [Guide](docs/validators/README.md) |

The [documentation index](docs/README.md) links API references, runnable
examples, and Colab tutorials. Cross-module workflows live in
[`examples/integration`](examples/integration) with supporting guides in
[`docs/integration`](docs/integration/README.md).

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
