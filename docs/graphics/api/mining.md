# `graphics.mining`

```python
graphics.mining(mining_rows, *, kind='counts', title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot positive/background breakdown or confidence bars across mining strategies.

```python
from klygo import graphics

rows = [{"strategy": "S3", "positive_images": 93, "background_images": 157, "boxes": 210, "mean_conf": 0.80, "median_conf": 0.82}]
graphics.mining(rows, kind="counts", show=False)
```

See [`examples/graphics/mining.py`](../../../examples/graphics/mining.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
