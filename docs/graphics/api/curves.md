# `graphics.curves`

```python
graphics.curves(grouped_runs, *, metric='box_loss', split='train', band=True, per_seed=False, mosaic_marker='auto', title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot a single training or validation loss/metric curve across grouped runs.

```python
from klygo import graphics

grouped = {"groups": {"S1": {"runs": [], "history_mean": {"epoch": [1, 2], "train/box_loss": [1.5, 1.1]}, "history_std": {"train/box_loss": [0.1, 0.05]}}}}
graphics.curves(grouped, metric="box_loss", show=False)
```

See [`examples/graphics/curves.py`](../../../examples/graphics/curves.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
