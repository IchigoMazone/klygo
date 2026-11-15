# `mining.compare`

```python
compare(before, after, *, quality=False, verbose=False)
```

Compares two materialized datasets and returns their summaries plus changes in image count, box count, class count, and imbalance ratio.

```python
from klygo import mining, utils
comparison = mining.compare(data, selected)
utils.tabulate(comparison.to_dict())
```

See [`examples/mining/compare.py`](../../../examples/mining/compare.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
