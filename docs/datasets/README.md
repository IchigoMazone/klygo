# `klygo.datasets` Documentation

`klygo.datasets` provides eight focused APIs for loading, inspecting, restructuring,
combining, relabeling, and splitting YOLO-style datasets.

Colab tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/datasets/Document_Datasets_Overview.ipynb)

The implementation shares the established Klygo infrastructure:

- `klygo.files` owns all filesystem operations.
- `klygo.config` reads and writes `data.yaml`.
- `klygo.archive` extracts and creates archives.
- `klygo.media` loads and saves images.
- `klygo.outputs.detect` supplies the normalized dataset results.

No dataset operation calls `shutil`, `zipfile`, or format parsers directly.
Temporary extraction and workspace directories are isolated and removed on
success or failure.
Outputs are prepared in a temporary location and replace existing destinations
only after successful completion. ZIP metadata omits machine-specific absolute
paths, so archives remain usable after moving or extracting them.

## API overview

| Task | API | Result |
|---|---|---|
| Flatten and load all images and annotations | [`load`](api/load.md) | `Detections` |
| Inspect classes, labels, and splits | [`info`](api/info.md) | `dict` |
| Create train/val/test folders | [`partition`](api/partition.md) | `Path` |
| Replace existing split ratios | [`repartition`](api/repartition.md) | `Path` |
| Flatten split folders | [`unpartition`](api/unpartition.md) | `Path` |
| Combine datasets and class IDs | [`merge`](api/merge.md) | ZIP `Path` |
| Produce multiple class/ratio archives | [`split`](api/split.md) | `list[Path]` |
| Rename classes or move IDs | [`remap_classes`](api/remap_classes.md) | `Path` |

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
Paths accept strings, `pathlib.Path`, and other text-returning `os.PathLike`
objects. When nested image paths are flattened, directory separators are
encoded in filenames to prevent collisions.
Sparse numeric class IDs in `data.yaml` are preserved across inspection, merge,
split, remap, and unpartition operations.

## Load the whole dataset

```python
from klygo import datasets

flat = datasets.unpartition("dataset.zip", "flat")
results = datasets.load(flat)
results[0].show()
crops = results.crop()
```

`unpartition()` creates the concrete flat dataset; `load()` then reads its images
and YOLO detection annotations into one `Detections` collection. It loads RGB
pixels into memory, so viewing and cropping remain available after return.
Memory use scales with the decoded dataset.
Images without labels are retained with zero boxes. Ground-truth boxes have
score `1.0`; original class IDs are in box metadata, and relative image paths
and split names are in detection metadata.

## Typical workflow

```python
from klygo import datasets

stats = datasets.info("raw.zip")
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

## Examples and tests

- Executable examples: [`examples/datasets/info.py`](../../examples/datasets/info.py)
- Behavioral tests: [`test/datasets/test_api.py`](../../test/datasets/test_api.py)
- Loading tests: [`test/datasets/test_load.py`](../../test/datasets/test_load.py)
- Documentation contract: [`audit_public_api.py`](../../.agents/skills/klygo-module-guides/scripts/audit_public_api.py)
