# `graphics.plot_partition_distribution`

```python
graphics.plot_partition_distribution(summary, *, title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot image counts and percentages across dataset partitions.

```python
from klygo import graphics

graphics.plot_partition_distribution([{"split": "train", "images": 300, "ratio_pct": 50.0}, {"split": "val", "images": 300, "ratio_pct": 50.0}], show=False)
```

See [`examples/graphics/plot_partition_distribution.py`](../../../examples/graphics/plot_partition_distribution.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
