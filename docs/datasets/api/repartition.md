# `datasets.repartition`

```python
datasets.repartition(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    ratios: tuple[float, ...],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> pathlib.Path
```

Flatten existing dataset splits and rebuild `train`, `val`, and `test` partitions with new deterministic ratios.

## Contract

Flattens any existing `train`/`val`/`test` split hierarchy in `source` (encoding relative paths into filenames so duplicate basenames across splits never collide) and redistributes all image/label pairs into new `train`, `val`, and `test` folders according to `ratios`.

Uses the same deterministic partitioning engine as [`datasets.partition()`](partition.md) and writes portable `data.yaml` metadata when targeting a `.zip` archive.

## Parameters

- `source`: `str | os.PathLike[str]` — partitioned dataset directory or supported archive.
- `target`: `str | os.PathLike[str]` — destination dataset directory or `.zip` archive path.
- `ratios`: `tuple[float, ...]` — 1, 2, or 3 positive split ratios (see [`datasets.partition()`](partition.md)).
- `format`: `str`, default `"yolo"` — dataset format to repartition. Currently only `"yolo"` is supported.
- `overwrite`: `bool`, default `False` — replace `target` if it already exists.
- `verbose`: `bool`, default `True` — display progress when creating a `.zip` target.

## Returns

`pathlib.Path` pointing to the newly repartitioned dataset directory or `.zip` archive.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `ValueError` if `format` is unsupported or `ratios` are invalid.
- Raises `FileNotFoundError` if `source` or its `images` directory is missing.
- Raises `FileExistsError` if `target` already exists and `overwrite=False`.

## AI usage guidance

Prefer `datasets.repartition()` over `datasets.partition()` when the input dataset already has `train`/`val`/`test` folders and you want to rebalance split proportions without creating nested `train/train` directories.

## Example

See [`repartition.py`](../../../examples/datasets/repartition.py) for an executable example.

## Tests

See [`test_partition.py`](../../../test/datasets/test_partition.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.repartition."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import archive, config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    for idx, split_name in enumerate(("train", "train", "val", "val")):
        media.save(source / "images" / split_name / f"img_{idx}.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(source / "labels" / split_name / f"img_{idx}.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(
        source / "data.yaml",
        {"train": "images/train", "val": "images/val", "nc": 1, "names": ["apple"]},
        overwrite=True,
        verbose=False,
    )

    output = datasets.repartition(source, root / "repartitioned.zip", (0.5, 0.25, 0.25), verbose=False)
    assert archive.is_archive(output)
    assert datasets.info(output)["total_images"] == 4
```
