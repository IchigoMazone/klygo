# `mining.deduplicate`

```python
deduplicate(data, *, keep="best", threshold=0.95, verbose=False)
```

Returns an independent dataset with duplicate frames removed. `keep` accepts `best`, `first`, `quality`, or `annotations`. The input is unchanged.

```python
from klygo import mining
unique = mining.deduplicate(data, keep="best")
```

See [`examples/mining/deduplicate.py`](../../../examples/mining/deduplicate.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
