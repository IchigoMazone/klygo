# `graphics.save_figure`

```python
graphics.save_figure(fig, path, *, dpi=300, tight=True)
```

Save a Matplotlib figure to disk in PNG, PDF, or SVG format.

```python
from tempfile import TemporaryDirectory
from klygo import files, graphics

fig = graphics.plot_partition_distribution([{"split": "train", "images": 10, "ratio_pct": 100.0}], show=False, return_figure=True)
with TemporaryDirectory() as tmp:
    out = graphics.save_figure(fig, files.path(tmp) / "chart.png")
    print(files.is_file(out))
```

See [`examples/graphics/save_figure.py`](../../../examples/graphics/save_figure.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
