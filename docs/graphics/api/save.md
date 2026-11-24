# `graphics.save`

```python
graphics.save(fig, path, *, dpi=300, tight=True)
```

Save a Matplotlib figure to disk in PNG, PDF, or SVG format.

```python
from tempfile import TemporaryDirectory
from klygo import files, graphics

fig = graphics.partitions([{"split": "train", "images": 10, "ratio_pct": 100.0}], show=False, return_figure=True)
with TemporaryDirectory() as tmp:
    out = graphics.save(fig, files.path(tmp) / "chart.png")
    print(files.is_file(out))
```

See [`examples/graphics/save.py`](../../../examples/graphics/save.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
