# `datasets.remap_classes`

```python
datasets.remap_classes(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    class_map: dict[int | str, int | str],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> pathlib.Path
```

Rename classes or reassign their numeric class IDs across `data.yaml` and all YOLO label files.

## Contract

Copies all dataset images and rewrites `.txt` annotation files while updating `data.yaml` according to `class_map`:
- **String keys** (`"car"`) select a class by its current name; **integer keys** (`1`) select a class by its current numeric ID.
- **String values** (`"vehicle"`) rename the selected class; **integer values** (`4`) change its output numeric ID.
- When resulting class IDs are contiguous (`0..N-1`), `data.yaml` stores `names` as a list; when IDs are sparse, `data.yaml` stores `names` as an ID-to-name mapping and sets `nc` to `max(id) + 1`.

## Parameters

- `source`: `str | os.PathLike[str]` — input dataset directory or supported archive.
- `target`: `str | os.PathLike[str]` — destination dataset directory or `.zip` archive path.
- `class_map`: `dict[int | str, int | str]` — mapping of current class names or IDs to new class names or non-negative integer IDs.
- `format`: `str`, default `"yolo"` — dataset format to remap. Currently only `"yolo"` is supported.
- `overwrite`: `bool`, default `False` — replace `target` if it already exists.
- `verbose`: `bool`, default `True` — display progress when creating a `.zip` target.

## Returns

`pathlib.Path` pointing to the created dataset directory or `.zip` archive.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str` or if `class_map` keys/values are not `int` or `str` (booleans are rejected).
- Raises `ValueError` if `format` is unsupported, if a key refers to an unknown class name or out-of-range class ID, if a target ID is negative, or if two different class names map to the same destination ID.
- Raises `FileNotFoundError` if `source` or its `images` directory does not exist.
- Raises `FileExistsError` if `target` already exists and `overwrite=False`.

## AI usage guidance

Each entry in `class_map` performs either a rename (`-> str`) or an ID reassignment (`-> int`). If a class needs both a new name and a new numeric ID, chain two `datasets.remap_classes()` calls or map by name and ID in sequence.

## Example

See [`remap_classes.py`](../../../examples/datasets/remap_classes.py) for an executable example.

## Tests

See [`test_remap.py`](../../../test/datasets/test_remap.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.remap_classes."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    media.save(source / "images" / "sample_0.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
    files.save(source / "labels" / "sample_0.txt", ["0 0.5 0.5 0.5 0.5", "1 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(source / "data.yaml", {"train": "images", "nc": 2, "names": ["apple", "banana"]}, overwrite=True, verbose=False)

    output = datasets.remap_classes(source, root / "remapped", {"apple": "fruit", 1: 3}, verbose=False)
    metadata = config.load(output / "data.yaml", verbose=False)
    assert metadata.names == {0: "fruit", 3: "banana"}
    assert datasets.info(output)["nc"] == 4
```
