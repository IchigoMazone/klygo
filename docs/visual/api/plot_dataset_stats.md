# `visual.plot_dataset_stats`

```python
visual.plot_dataset_stats(source, figsize=(10, 6))
```

Plot the number of labeled objects per class and return `None`. `source` is a
dataset directory or archive path; `figsize` controls Matplotlib figure inches.
The function reads class names from dataset metadata and counts label rows under
`images` and `labels`. Archives are extracted to a temporary directory. Invalid
sources or unavailable plotting dependencies raise their underlying errors.

```python
import tempfile
import matplotlib
from PIL import Image
from klygo import files, visual

matplotlib.use("Agg", force=True)
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    files.mkdir(root / "images")
    files.mkdir(root / "labels")
    files.save(root / "data.yaml", "names: [cat]\nnc: 1\n")
    Image.new("RGB", (8, 8)).save(root / "images" / "sample.png")
    files.save(root / "labels" / "sample.txt", "0 0.5 0.5 0.2 0.2\n")
    visual.plot_dataset_stats(root)
```

See [the runnable example](../../../examples/visual/plot_dataset_stats.py) and
[`show_image`](show_image.md) for image display.
