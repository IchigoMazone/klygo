# `datasets.split`

```python
datasets.split(
    source: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    format: str = "yolo",
    by_class: bool = False,
    class_groups: list[list[str]] | None = None,
    ratios: tuple[float, ...] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[pathlib.Path]
```

Split a YOLO dataset into multiple standalone `.zip` archives by individual class, explicit class groups, or deterministic image ratios.

## Contract

Selects exactly one splitting strategy:
1. `by_class=True`: creates one `.zip` archive per class (`split_<class>.zip`) containing only images and annotations matching that class, remapping the kept class ID to `0`.
2. `class_groups=[...]`: creates one `.zip` archive per named group (`split_<class>.zip` for single-class groups or `split_group_<i>.zip` for multi-class groups), filtering annotations to the group and remapping class IDs to `0..len(group)-1`.
3. `ratios=(...)`: shuffles all image/label pairs deterministically (seed `42`) and writes `split_part_<i>.zip` archives preserving the original class map (including sparse IDs).

## Parameters

- `source`: `str | os.PathLike[str]` — input dataset directory or supported archive.
- `output_dir`: `str | os.PathLike[str]` — directory where output `.zip` archives will be created.
- `format`: `str`, default `"yolo"` — dataset format to split. Currently only `"yolo"` is supported.
- `by_class`: `bool`, default `False` — create one archive per class defined in `data.yaml`.
- `class_groups`: `list[list[str]] | None`, default `None` — explicit groups of class names for group-based archives.
- `ratios`: `tuple[float, ...] | None`, default `None` — two or more positive floats summing to `1.0` for ratio-based archives.
- `overwrite`: `bool`, default `False` — replace existing split archives in `output_dir`.
- `verbose`: `bool`, default `True` — display archive creation progress.

## Returns

`list[pathlib.Path]` of created `.zip` archives in deterministic order. (Class/group archives with zero matching images are omitted.)

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `ValueError` if `format` is unsupported, if zero or more than one split strategy (`by_class`, `class_groups`, `ratios`) is provided, if `class_groups` contains empty groups or unknown class names, or if `ratios` has fewer than 2 items or does not sum to `1.0`.
- Raises `FileNotFoundError` if `source` or its `images` directory is missing.
- Raises `FileExistsError` if a destination `.zip` archive already exists and `overwrite=False`.

## AI usage guidance

Use `datasets.split()` when you need separate, self-contained `.zip` dataset archives (e.g., per-class sub-datasets or sharded archives). To create `train`/`val`/`test` folders inside a single dataset, use [`datasets.partition()`](partition.md) instead.

## Example

See [`split.py`](../../../examples/datasets/split.py) for an executable example.

## Tests

See [`test_split.py`](../../../test/datasets/test_split.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.split."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import archive, config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    for idx in range(4):
        media.save(source / "images" / f"img_{idx}.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(source / "labels" / f"img_{idx}.txt", [f"{idx % 2} 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(source / "data.yaml", {"train": "images", "nc": 2, "names": ["apple", "banana"]}, overwrite=True, verbose=False)

    parts = datasets.split(source, root / "parts", ratios=(0.5, 0.5), verbose=False)
    assert len(parts) == 2
    assert all(archive.is_archive(path) for path in parts)
    assert sum(datasets.info(path)["total_images"] for path in parts) == 4
```
