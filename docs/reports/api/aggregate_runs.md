# `reports.aggregate_runs`

```python
reports.aggregate_runs(grouped_runs, *, format='yolo', best='auto', metric_names=('precision', 'recall', 'f1', 'mAP50', 'mAP50-95'), compare_pair=None)
```

Aggregate multi-seed training runs into summary tables and epoch curves.

```python
from klygo import reports

run_a = {"config": {"seed": 42}, "best_epoch": 5, "total_time_s": 100.0, "best_metrics": {"mAP50-95": 0.65}, "history": {"epoch": [1], "train/box_loss": [1.0]}}
run_b = {"config": {"seed": 100}, "best_epoch": 5, "total_time_s": 102.0, "best_metrics": {"mAP50-95": 0.67}, "history": {"epoch": [1], "train/box_loss": [0.9]}}
print(reports.aggregate_runs({"S1": [run_a, run_b]})["table"])
```

See [`examples/reports/aggregate_runs.py`](../../../examples/reports/aggregate_runs.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
