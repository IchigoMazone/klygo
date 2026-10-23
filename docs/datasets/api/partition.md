# `datasets.partition`

```python
datasets.partition(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    ratios: tuple[float, ...],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> pathlib.Path
```

Partition a YOLO dataset reproducibly into `train`, `val`, and optional `test` directories or a `.zip` archive.

## Contract

Scans all image/label pairs in `source`, shuffles them with a deterministic seed, and distributes them into `images/{train,val,test}` and `labels/{train,val,test}` according to `ratios`. Writes an updated `data.yaml` and publishes the result to `target` (either a directory or a `.zip` archive).

## Parameters

- `source`: `str | os.PathLike[str]` — source dataset directory or supported archive.
- `target`: `str | os.PathLike[str]` — destination dataset directory or `.zip` archive path.
- `ratios`: `tuple[float, ...]` — 1, 2, or 3 positive floats:
  - 1 float `(train,)`: splits remainder equally between `val` and `test`.
  - 2 floats `(train, val)`: assigns any remaining fraction to `test` (or creates only `train` and `val` when they sum to `1.0`).
  - 3 floats `(train, val, test)`: must sum to `1.0`.
- `format`: `str`, default `"yolo"` — dataset format to partition. Currently only `"yolo"` is supported.
- `overwrite`: `bool`, default `False` — replace `target` if it already exists.
- `verbose`: `bool`, default `True` — display progress when writing a `.zip` target.

## Returns

`pathlib.Path` pointing to the created dataset directory or `.zip` archive.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `ValueError` if `format` is unsupported or if `ratios` is empty, has more than 3 values, contains non-finite or out-of-range numbers, or does not sum to `1.0` when 3 values are given.
- Raises `FileNotFoundError` if `source` or its `images` directory does not exist.
- Raises `FileExistsError` if `target` already exists and `overwrite=False`.

## AI usage guidance

Use `datasets.partition()` when structuring an unpartitioned dataset into `train`/`val`/`test` splits. When replacing splits on an already partitioned dataset, prefer [`datasets.repartition()`](repartition.md) to make the intent explicit.

## Example

See [`partition.py`](../../../examples/datasets/partition.py) for an executable example.

## Tests

See [`test_partition.py`](../../../test/datasets/test_partition.py) and [`test_api.py`](../../../test/datasets/test_api.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.partition."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    for idx in range(4):
        media.save(source / "images" / f"img_{idx}.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
        files.save(source / "labels" / f"img_{idx}.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(source / "data.yaml", {"train": "images", "nc": 1, "names": ["apple"]}, overwrite=True, verbose=False)

    output = datasets.partition(source, root / "partitioned", (0.5, 0.25, 0.25), verbose=False)
    assert files.is_dir(output / "images" / "train")
    assert datasets.info(output)["splits"] == {"train": 2, "val": 1, "test": 1}
```
