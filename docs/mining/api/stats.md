# `mining.stats`

```python
stats(data, *, quality=True, cooccurrence=True)
```

Returns a [`MiningStats`](MiningStats.md) report for materialized `Detections`. Class counts use both images and bounding boxes. Quality metrics decode source images only when `quality=True`.

```python
from klygo import mining, utils
report = mining.stats(data, quality=True)
utils.tabulate(report.to_dict())
```

See [`examples/mining/stats.py`](../../../examples/mining/stats.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
