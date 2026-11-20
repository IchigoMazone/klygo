# `mining.score`

```python
score(data, *, weights=None, verbose=False)
```

Returns one normalized score per image. `weights` may contain `quality`, `rarity`, `coverage`, and `boxes` (all finite non-negative numbers summing to `> 0`); `weights=None` uses the default blend (`quality=0.4`, `rarity=0.3`, `coverage=0.2`, `boxes=0.1`), whereas passing `{}` raises `ValueError`. All six quality metrics (`brightness`, `sharpness`, `contrast`, `entropy`, `1 - noise`, `exposure`) are min-max normalized onto `[0, 1]`, and `coverage` is measured against classes actually present in the dataset.

```python
from klygo import mining
scores = mining.score(data, weights={"quality": 0.6, "rarity": 0.4})
```

See [`examples/mining/score.py`](../../../examples/mining/score.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
