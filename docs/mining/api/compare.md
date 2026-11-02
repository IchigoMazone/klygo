# `mining.compare`

```python
compare(before, after, *, quality=False)
```

Compares two materialized datasets and returns their summaries plus changes in image count, box count, class count, and imbalance ratio.

```python
from klygo import mining
comparison = mining.compare(data, selected)
comparison.show()
```

See [`examples/mining/compare.py`](../../../examples/mining/compare.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
