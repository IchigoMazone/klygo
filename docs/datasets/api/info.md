# `datasets.info`

```python
datasets.info(
    source: str | os.PathLike[str],
    format: str = "yolo",
    verbose: bool = False,
) -> dict[str, Any]
```

Inspect structural statistics, class metadata, and split counts for a YOLO dataset directory or archive without modifying it.

## Contract

Reads `data.yaml`, scans the `images` and `labels` directories (extracting to an isolated temporary workspace when `source` is a supported archive), and counts labeled and unlabeled images across `train`, `val`, `test`, or `raw` splits.

When `verbose=True`, prints a formatted Rich summary table to the console before returning the statistics dictionary.

## Parameters

- `source`: `str | os.PathLike[str]` — dataset directory or supported archive containing `data.yaml` and `images/`.
- `format`: `str`, default `"yolo"` — dataset annotation format to inspect. Currently only `"yolo"` is supported.
- `verbose`: `bool`, default `False` — print the dataset statistics as a Rich table while returning the dictionary.

## Returns

`dict[str, Any]` containing:
- `"classes"`: `list[str]` — ordered class names in ascending class ID order.
- `"class_ids"`: `list[int]` — numeric class IDs corresponding to `"classes"`.
- `"nc"`: `int` — class count (`max(class_ids) + 1`, or `0` when empty), preserving sparse ID capacity.
- `"total_images"`: `int` — total number of discovered dataset images.
- `"total_labels"`: `int` — number of images with a matching `.txt` label file.
- `"unlabeled_images"`: `int` — number of images without a matching `.txt` label file.
- `"splits"`: `dict[str, int]` — image counts grouped by `"train"`, `"val"`, `"test"`, or `"raw"`.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `ValueError` if `format` is unsupported or if `data.yaml` has missing, duplicate, or invalid class names/IDs.
- Raises `FileNotFoundError` if `source` or its `images` directory does not exist.

## AI usage guidance

Call `datasets.info()` before running destructive or expensive transformations (`partition`, `merge`, `remap_classes`, `load`) to verify class names, sparse class IDs, and whether the dataset is already partitioned (`splits` keys in `{"train", "val", "test"}`) or flat (`"raw"`).

## Example

See [`info.py`](../../../examples/datasets/info.py) for an executable example.

## Tests

See [`test_info.py`](../../../test/datasets/test_info.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.info."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory) / "source"
    media.save(root / "images" / "train" / "sample_0.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
    files.save(root / "labels" / "train" / "sample_0.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(
        root / "data.yaml",
        {"train": "images/train", "nc": 2, "names": ["apple", "banana"]},
        overwrite=True,
        verbose=False,
    )

    stats = datasets.info(root, format="yolo", verbose=False)
    assert stats["classes"] == ["apple", "banana"]
    assert stats["total_images"] == 1
    assert stats["splits"] == {"train": 1}
```
