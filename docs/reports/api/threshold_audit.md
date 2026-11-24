# `reports.threshold_audit`

```python
reports.threshold_audit(pred_calib, target_calib, *, thresholds=None, iou=0.5, pred_confirm=None, target_confirm=None, verbose=False)
```

Sweep confidence thresholds on calibration data and select optimal `T*`.

```python
from klygo import outputs, reports

pred = outputs.Detections([outputs.Detection([outputs.Box((0, 0, 20, 20), label="hat", score=0.9)], source_id="a", image_size=(64, 64))])
target = outputs.Detections([outputs.Detection([outputs.Box((0, 0, 20, 20), label="hat")], source_id="a", image_size=(64, 64))])
print(reports.threshold_audit(pred, target, thresholds=[0.3, 0.5])["best_threshold"])
```

See [`examples/reports/threshold_audit.py`](../../../examples/reports/threshold_audit.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
