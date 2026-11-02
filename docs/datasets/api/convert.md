# `datasets.convert`

```python
convert(source, target, format="yolo", tree=None, classes=None, on_conflict="error", overwrite=False)
```

Convert one or more mapped image and label directory trees into a flat YOLO dataset containing `images/`, `labels/`, and `data.yaml`. The first `tree` value selects image directories, the second selects matching label directories, and an optional third value selects exactly one YAML metadata file. Key names do not affect their roles.

Image and label pattern lists are paired by position. Glob matches are sorted before corresponding directories are paired, preventing annotations with the same stem in different batches from being mixed.

`on_conflict` applies to complete image-label samples:

- `error` raises before output is published.
- `rename` keeps both samples with a synchronized numeric suffix.
- `skip` keeps the first sample.
- `overwrite` keeps the last sample.

```python
from klygo import datasets

flat = datasets.convert(
    "raw_dataset",
    "flat_dataset",
    tree={
        "pictures": "batch_*/images",
        "annotations": "batch_*/labels",
    },
    classes={0: "apple", 1: "banana"},
    on_conflict="rename",
)
datasets.partition(flat, "partitioned", (0.8, 0.1, 0.1))
```

The conversion is staged and atomically published. `overwrite` controls a pre-existing target; it is independent from conflicts among inputs. Images without labels are retained as negative samples. Orphan labels are ignored with a warning.

Raises `ValueError` for invalid tree structure, overlapping source and target paths, ambiguous glob groups, conflicting metadata, non-contiguous classes, unknown label IDs, or sample conflicts. Raises `FileExistsError` when the target exists without `overwrite=True`.

See [`examples/datasets/convert.py`](../../../examples/datasets/convert.py) and [`test/datasets/test_convert.py`](../../../test/datasets/test_convert.py).
