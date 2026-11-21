# `graphics.bars`

```python
graphics.bars(rows, *, metrics=('mAP50-95',), error_bars=True, show_values=None, title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot single-metric bars with optional std error bars or multi-metric grouped bars. Pass `show_values=False` to always hide numeric labels above bars, `show_values=True` to force displaying them, or `None` (default) to automatically show them only on single-metric plots with at most 15 items.

```python
from klygo import graphics

rows = [{"strategy": "S1", "mAP50-95_mean": 0.65, "mAP50-95_std": 0.01}, {"strategy": "S4", "mAP50-95_mean": 0.68, "mAP50-95_std": 0.006}]
graphics.bars(rows, show=False)
```

See [`examples/graphics/bars.py`](../../../examples/graphics/bars.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
