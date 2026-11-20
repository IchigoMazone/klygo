# `reports.rank_evaluations`

```python
reports.rank_evaluations(evaluations, *, sort_by='mAP50-95', baseline=None)
```

Rank evaluation results descending by `sort_by` and compute baseline deltas.

```python
from klygo import reports

print(reports.rank_evaluations([{"code": "S1", "mAP50-95": 0.57}, {"code": "S4", "mAP50-95": 0.60}], baseline="S1"))
```

See [`examples/reports/rank_evaluations.py`](../../../examples/reports/rank_evaluations.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
