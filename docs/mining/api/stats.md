# `mining.stats`

```python
stats(data, *, quality=True, cooccurrence=True, verbose=False)
```

Returns a [`MiningStats`](MiningStats.md) report for materialized `Detections`. Class counts use both images and bounding boxes. Quality metrics decode source images only when `quality=True`.

```python
from klygo import mining, utils
report = mining.stats(data, quality=True)
utils.tabulate(report.to_dict())
```

See [`examples/mining/stats.py`](../../../examples/mining/stats.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
