# `mining.find_duplicates`

```python
find_duplicates(data, *, exact=True, method="phash", threshold=0.95)
```

Groups duplicate images anywhere in a materialized dataset. Exact matching uses file checksums and near matching uses a cached perceptual hash. This function reports matches and never removes data.

```python
from klygo import mining
duplicates = mining.find_duplicates(data, threshold=0.97)
print(duplicates.groups)
```

See [`examples/mining/find_duplicates.py`](../../../examples/mining/find_duplicates.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
