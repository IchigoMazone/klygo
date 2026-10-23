# `datasets.load`

```python
datasets.load(
    source: str | os.PathLike[str],
    format: str = "yolo",
    verbose: bool = True,
) -> klygo.outputs.detect.Detections
```

Load all images and five-column YOLO detection annotations from a flat dataset directory or archive into a `Detections` collection.

## Contract

Reads a flat YOLO dataset (`images/<file>` and optional `labels/<file>.txt` alongside `data.yaml`), decodes each image into an in-memory RGB PIL image via `klygo.media`, and converts normalized `[class_id, cx, cy, width, height]` rows into pixel XYXY `Box` instances.

For a partitioned dataset (`train`/`val`/`test` folders), call [`datasets.unpartition()`](unpartition.md) first and pass the resulting flat directory or `.zip` archive to `datasets.load()`. The input dataset is never modified.

## Parameters

- `source`: `str | os.PathLike[str]` — flat YOLO dataset directory or supported archive containing `data.yaml`, `images/`, and optional `labels/`.
- `format`: `str`, default `"yolo"` — annotation format to decode. Currently only `"yolo"` is supported.
- `verbose`: `bool`, default `True` — display a progress bar while loading images and annotations.

## Returns

`klygo.outputs.detect.Detections` with `source_type="images"`. Each `Detection` item contains:
- An in-memory RGB PIL `source_image` (safe to `.show()` or `.crop()` even after temporary workspaces are removed).
- `Box` objects in pixel `[x1, y1, x2, y2]` coordinates with `score=1.0`, `box.metadata["class_id"]` preserving the original numeric ID (including sparse IDs), and `box.metadata["ground_truth"] = True`.
- Detection metadata fields: `dataset_source`, `relative_path` (decoded back to the original split-relative path when flattened by `unpartition()`), `split` (`"train"`, `"val"`, `"test"`, or `"raw"`), and `ground_truth=True`.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `FileNotFoundError` if `source` or its `images` directory is missing.
- Raises `ValueError` if `format` is unsupported, if the dataset is partitioned (must call `datasets.unpartition()` first), if no supported images exist, or if any label line violates the five-column normalized YOLO detection contract (`class_id cx cy width height` with coordinates in `[0, 1]` and positive width/height).
- Raises `OSError` if an image file is corrupt or unreadable.
- Missing or empty `.txt` label files produce `Detection` objects with zero boxes.

## AI usage guidance

Always flatten partitioned datasets with `datasets.unpartition()` before calling `datasets.load()`. Because `load()` materializes RGB pixels in memory for every image, use [`datasets.info()`](info.md) instead when you only need counts or class metadata.

## Example

See [`load.py`](../../../examples/datasets/load.py) for an executable example.

## Tests

See [`test_load.py`](../../../test/datasets/test_load.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.load."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    media.save(source / "images" / "train" / "sample_0.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
    files.save(source / "labels" / "train" / "sample_0.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(
        source / "data.yaml",
        {"train": "images/train", "nc": 1, "names": ["apple"]},
        overwrite=True,
        verbose=False,
    )

    flat = datasets.unpartition(source, root / "flat", verbose=False)
    results = datasets.load(flat, format="yolo", verbose=False)
    assert len(results) == 1
    assert results.total_objects == 1
    assert results[0].metadata["split"] == "train"
```
