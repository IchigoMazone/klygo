# `graphics.partitions`

```python
graphics.partitions(summary, *, title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot image counts and percentages across dataset partitions.

```python
from klygo import graphics

graphics.partitions([{"split": "train", "images": 300, "ratio_pct": 50.0}, {"split": "val", "images": 300, "ratio_pct": 50.0}], show=False)
```

See [`examples/graphics/partitions.py`](../../../examples/graphics/partitions.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
