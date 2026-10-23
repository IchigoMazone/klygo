# `visual.plot_dataset_stats`

```python
visual.plot_dataset_stats(
    source,
    figsize=(10, 6),
)
```

Plot YOLO object counts for each class in a dataset directory or archive.

## Contract

Inspects a YOLO dataset located at `source` (either an extracted directory or a supported archive file such as `.zip` or `.tar.gz`) and renders a Matplotlib bar chart of object instance counts per class.

First, `datasets.info(source)` is called to retrieve the dataset's `classes` list and `class_ids`. If `source` points to a regular file, it is extracted via `archive.extract()` into a temporary directory that is automatically cleaned up after scanning. The function locates `images/` and `labels/` subdirectories, scans every paired label file, parses the leading integer class ID on each non-empty line, and increments the count for known `class_ids`. Finally, it creates a Matplotlib figure of size `figsize` with `"Classes"` on the x-axis, `"Object Count"` on the y-axis, and title `"Class Distribution in Dataset"`, and calls `plt.show()`.

## Parameters

- `source`: `str | pathlib.Path` — path to a YOLO dataset root directory or a supported dataset archive file.
- `figsize`: `tuple[int, int]`, default `(10, 6)` — Matplotlib figure size `(width_inches, height_inches)`.

## Returns

`None` — renders and displays the class distribution bar chart via `matplotlib.pyplot.show()`.

## Errors and edge cases

- Raises `FileNotFoundError` or `ValueError` (propagated from `klygo.datasets.info` or `klygo.archive.extract`) when `source` does not exist or is not a valid dataset directory/archive.
- Malformed label lines whose first token cannot be parsed as an `int` or whose class ID is not declared in dataset metadata are silently skipped without aborting the scan.
- If `images/` or `labels/` directories are absent, all declared classes are plotted with a count of `0`.

## AI usage guidance

Prefer `from klygo import visual` followed by `visual.plot_dataset_stats(...)` to visually audit class balance before training or after running `klygo.datasets` transformations.

## Example

See [`plot_dataset_stats.py`](../../../examples/visual/plot_dataset_stats.py) for an executable example.

## Tests

See [`test_visual.py`](../../../test/visual/test_visual.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.visual.plot_dataset_stats."""

import tempfile
import matplotlib
from PIL import Image
from klygo import files, visual

matplotlib.use("Agg", force=True)
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    files.mkdir(root / "images")
    files.mkdir(root / "labels")
    files.save(root / "data.yaml", "names: [cat, dog]\nnc: 2\n", verbose=False)
    Image.new("RGB", (8, 8)).save(root / "images" / "sample.png")
    files.save(
        root / "labels" / "sample.txt",
        "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
        verbose=False,
    )
    result = visual.plot_dataset_stats(root, figsize=(8, 4))
    print("Plotted two class counts")
    assert result is None
```
