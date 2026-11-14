# `mining.compare`

```python
compare(before, after, *, quality=False)
```

Compares two materialized datasets and returns their summaries plus changes in image count, box count, class count, and imbalance ratio.

```python
from klygo import mining, utils
comparison = mining.compare(data, selected)
utils.tabulate(comparison.to_dict())
```

See [`examples/mining/compare.py`](../../../examples/mining/compare.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
