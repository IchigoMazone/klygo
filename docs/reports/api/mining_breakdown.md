# `reports.mining_breakdown`

```python
reports.mining_breakdown(subsets)
```

Summarize positive/background split, box density, and confidence across mining subsets.

```python
from klygo import outputs, reports

sub = outputs.Detections([outputs.Detection([outputs.Box((0, 0, 20, 20), label="hat", score=0.8)], source_id="a", image_size=(64, 64))])
print(reports.mining_breakdown({"S4": sub}))
```

See [`examples/reports/mining_breakdown.py`](../../../examples/reports/mining_breakdown.py) and [`test/reports/test_reports.py`](../../../test/reports/test_reports.py).
