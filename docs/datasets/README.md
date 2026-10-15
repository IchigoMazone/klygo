# `klygo.datasets` Documentation

`klygo.datasets` provides eight focused APIs for inspecting, restructuring,
combining, relabeling, splitting, and generating YOLO-style datasets.

The implementation shares the established Klygo infrastructure:

- `klygo.files` owns all filesystem operations.
- `klygo.config` reads and writes `data.yaml`.
- `klygo.archive` extracts and creates archives.
- `klygo.media` loads and saves images.
- `klygo.models` supplies normalized predictions to `datasets.export()`.

No dataset operation calls `shutil`, `zipfile`, or format parsers directly.
Temporary extraction and workspace directories are isolated and removed on
success or failure.

## API overview

| Task | API | Result |
|---|---|---|
| Inspect classes, labels, and splits | [`get_dataset_info`](api/get_dataset_info.md) | `dict` |
| Create train/val/test folders | [`partition`](api/partition.md) | `Path` |
| Replace existing split ratios | [`repartition`](api/repartition.md) | `Path` |
| Flatten split folders | [`unpartition`](api/unpartition.md) | `Path` |
| Combine datasets and class IDs | [`merge`](api/merge.md) | ZIP `Path` |
| Produce multiple class/ratio archives | [`split`](api/split.md) | `list[Path]` |
| Rename classes or move IDs | [`remap_classes`](api/remap_classes.md) | `Path` |
| Generate data from model results | [`export`](api/export.md) | directory `Path` |

## Dataset contract

Transformation APIs expect a YOLO-style root:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The root may be supplied directly or inside a supported archive. Nested roots
are discovered by locating `data.yaml` beside an `images` directory.

## Typical workflow

```python
from klygo import datasets

info = datasets.get_dataset_info("raw.zip")
partitioned = datasets.partition(
    "raw.zip",
    "partitioned",
    ratios=(0.8, 0.1, 0.1),
)
renamed = datasets.remap_classes(
    partitioned,
    "renamed.zip",
    {"automobile": "car"},
)
```

## Model export

`datasets.export()` consumes the current callable model API:

```python
generated = datasets.export(
    model,
    "generated",
    format="yolo",
    source="images",
    classes=["car", "person"],
    threshold=0.4,
)
```

It calls `model(images, prompt=..., batch=..., threshold=...)`. Classification
output is built from `Detection.crop()`; it does not require the removed
`model.crop()` API.

## Examples and tests

- Executable examples: [`examples/datasets`](../../examples/datasets/README.md)
- Behavioral tests: [`test/datasets/test_api.py`](../../test/datasets/test_api.py)
- Documentation contract: [`test_documentation.py`](test_documentation.py)
