# `datasets.unpartition`

```python
datasets.unpartition(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> pathlib.Path
```

Flatten `train`, `val`, and `test` split folders into a single `images/` and `labels/` directory or `.zip` archive.

## Contract

Copies all images and matching `.txt` label files from nested split directories into a flat `images/` and `labels/` layout under `target`. Relative split paths are percent/underscore-encoded into the flattened filenames so files with identical basenames in different splits (e.g., `train/001.jpg` and `val/001.jpg`) never collide and remain recoverable by [`datasets.load()`](load.md).

Preserves contiguous and sparse class ID mappings in `data.yaml`.

## Parameters

- `source`: `str | os.PathLike[str]` — partitioned dataset directory or supported archive.
- `target`: `str | os.PathLike[str]` — destination flat dataset directory or `.zip` archive path.
- `format`: `str`, default `"yolo"` — dataset format to unpartition. Currently only `"yolo"` is supported.
- `overwrite`: `bool`, default `False` — replace `target` if it already exists.
- `verbose`: `bool`, default `True` — display progress when creating a `.zip` target.

## Returns

`pathlib.Path` pointing to the created flat dataset directory or `.zip` archive.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `ValueError` if `format` is unsupported or `data.yaml` is invalid.
- Raises `FileNotFoundError` if `source` or its `images` directory does not exist.
- Raises `FileExistsError` if `target` already exists and `overwrite=False`.

## AI usage guidance

Always call `datasets.unpartition()` before [`datasets.load()`](load.md) when loading a partitioned `train`/`val`/`test` dataset into `Detections`.

## Example

See [`unpartition.py`](../../../examples/datasets/unpartition.py) for an executable example.

## Tests

See [`test_partition.py`](../../../test/datasets/test_partition.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.unpartition."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    for split_name in ("train", "val"):
        media.save(source / "images" / split_name / "sample.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(source / "labels" / split_name / "sample.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(
        source / "data.yaml",
        {"train": "images/train", "val": "images/val", "nc": 1, "names": ["apple"]},
        overwrite=True,
        verbose=False,
    )

    flat = datasets.unpartition(source, root / "flat", verbose=False)
    assert datasets.info(flat)["splits"] == {"raw": 2}
```
