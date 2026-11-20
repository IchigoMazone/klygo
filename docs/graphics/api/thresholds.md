# `graphics.thresholds`

```python
graphics.thresholds(audit_result, *, kind='curves', title=None, figsize=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Plot Precision/Recall/F1 curves or TP/FP stacked bars across thresholds.

```python
from klygo import graphics

audit = {"best_threshold": 0.55, "rows": [{"threshold": 0.3, "tp": 94, "fp": 21, "precision": 0.81, "recall": 0.76, "f1": 0.78}, {"threshold": 0.55, "tp": 88, "fp": 5, "precision": 0.94, "recall": 0.71, "f1": 0.81}]}
graphics.thresholds(audit, kind="curves", show=False)
```

See [`examples/graphics/thresholds.py`](../../../examples/graphics/thresholds.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
