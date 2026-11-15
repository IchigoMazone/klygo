# `mining.select`

```python
select(data, *, method="weighted", limit=None, **criteria, verbose=False)
```

Selects complete images and returns an independent `Detections`. Methods are `class`, `quality`, `boxes`, `random`, `quota`, `rare_first`, `coverage`, `diversity`, and `weighted`. Unknown class names raise an error instead of being ignored.

```python
from klygo import mining
cars = mining.select(data, method="class", classes=["car"], match="any")
diverse = mining.select(data, method="diversity", limit=100)
```

See [`examples/mining/select.py`](../../../examples/mining/select.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
