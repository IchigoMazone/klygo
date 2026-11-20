# `graphics.plot_benchmark_bars`

```python
graphics.plot_benchmark_bars(rows, *, metrics=('mAP50-95',), error_bars=True, title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot single-metric bars with optional std error bars or multi-metric grouped bars.

```python
from klygo import graphics

rows = [{"strategy": "S1", "mAP50-95_mean": 0.65, "mAP50-95_std": 0.01}, {"strategy": "S4", "mAP50-95_mean": 0.68, "mAP50-95_std": 0.006}]
graphics.plot_benchmark_bars(rows, show=False)
```

See [`examples/graphics/plot_benchmark_bars.py`](../../../examples/graphics/plot_benchmark_bars.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
