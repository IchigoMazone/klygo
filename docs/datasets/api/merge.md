# `datasets.merge`

```python
datasets.merge(
    sources: list[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> pathlib.Path
```

Combine two or more YOLO datasets into a single `.zip` archive while unifying class names and rewriting local class IDs.

## Contract

Opens each source dataset (directory or archive), builds a unified global class list in first-seen order across all sources, copies all images with source-index prefixes (`src0_...`, `src1_...`) and encoded relative paths to prevent filename collisions, and rewrites every `.txt` annotation file so local class IDs (including sparse IDs) map to the unified global class indices.

The merged dataset is staged in a temporary workspace and written as a portable `.zip` archive without machine-specific absolute paths in `data.yaml`.

## Parameters

- `sources`: `list[str | os.PathLike[str]]` — two or more dataset directories or supported archives to combine.
- `output_path`: `str | os.PathLike[str]` — destination `.zip` archive path.
- `format`: `str`, default `"yolo"` — dataset format to merge. Currently only `"yolo"` is supported.
- `overwrite`: `bool`, default `False` — replace `output_path` if it already exists.
- `verbose`: `bool`, default `True` — display archive creation progress.

## Returns

`pathlib.Path` pointing to the created `.zip` archive.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str` or `sources` is not a list/tuple of paths.
- Raises `ValueError` if `len(sources) < 2`, if `output_path` does not end with `.zip`, if `format` is unsupported, or if any source has invalid `data.yaml` metadata.
- Raises `FileNotFoundError` if any source dataset or its `images` directory is missing.
- Raises `FileExistsError` if `output_path` already exists and `overwrite=False`.

## AI usage guidance

Always use `datasets.merge()` rather than `archive.merge()` when combining YOLO datasets so `data.yaml` class names and `.txt` label IDs are reconciled semantically instead of overwriting files.

## Example

See [`merge.py`](../../../examples/datasets/merge.py) for an executable example.

## Tests

See [`test_merge.py`](../../../test/datasets/test_merge.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.merge."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    for name, classes in (("first", ["apple", "banana"]), ("second", ["banana", "orange"])):
        ds = root / name
        media.save(ds / "images" / f"{name}.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(ds / "labels" / f"{name}.txt", ["1 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
        config.save(ds / "data.yaml", {"train": "images", "nc": 2, "names": classes}, overwrite=True, verbose=False)

    merged = datasets.merge([root / "first", root / "second"], root / "merged.zip", verbose=False)
    info = datasets.info(merged)
    assert info["classes"] == ["apple", "banana", "orange"]
    assert info["total_images"] == 2
```
