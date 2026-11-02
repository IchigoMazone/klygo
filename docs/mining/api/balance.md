# `mining.balance`

```python
balance(data, *, target="median", strategy="select", count_by="images", seed=42)
```

Reduces image-level imbalance by selection or undersampling. `target` accepts an integer, `min`, `median`, or `mean`. Version 1 intentionally rejects oversampling because duplicated frame identities would violate the `Detections` identity contract.

```python
from klygo import mining
balanced = mining.balance(data, target="median", count_by="images")
```

See [`examples/mining/balance.py`](../../../examples/mining/balance.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
