# `mining.deduplicate`

```python
deduplicate(data, *, keep="best", threshold=0.95)
```

Returns an independent dataset with duplicate frames removed. `keep` accepts `best`, `first`, `quality`, or `annotations`. The input is unchanged.

```python
from klygo import mining
unique = mining.deduplicate(data, keep="best")
```

See [`examples/mining/deduplicate.py`](../../../examples/mining/deduplicate.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
