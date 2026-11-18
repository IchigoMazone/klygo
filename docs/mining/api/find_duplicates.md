# `mining.find_duplicates`

```python
find_duplicates(data, *, exact=True, method="phash", threshold=0.95, verbose=False)
```

Groups duplicate images anywhere in a materialized dataset. Exact matching uses file checksums and near matching uses a cached perceptual hash. This function reports matches and never removes data.

```python
from klygo import mining
duplicates = mining.find_duplicates(data, threshold=0.97)
print(duplicates.groups)
```

See [`examples/mining/find_duplicates.py`](../../../examples/mining/find_duplicates.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
