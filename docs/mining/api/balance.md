# `mining.balance`

```python
balance(data, *, target="median", strategy="select", count_by="images", seed=42, verbose=False)
```

Reduces image-level imbalance by selection or undersampling. `target` accepts an integer, `min`, `median`, or `mean`. Version 1 intentionally rejects oversampling because duplicated frame identities would violate the `Detections` identity contract.

```python
from klygo import mining
balanced = mining.balance(data, target="median", count_by="images")
```

See [`examples/mining/balance.py`](../../../examples/mining/balance.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
