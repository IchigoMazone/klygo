# `mining.select`

```python
select(data, *, method="weighted", limit=None, **criteria)
```

Selects complete images and returns an independent `Detections`. Methods are `class`, `quality`, `boxes`, `random`, `quota`, `rare_first`, `coverage`, `diversity`, and `weighted`. Unknown class names raise an error instead of being ignored.

```python
from klygo import mining
cars = mining.select(data, method="class", classes=["car"], match="any")
diverse = mining.select(data, method="diversity", limit=100)
```

See [`examples/mining/select.py`](../../../examples/mining/select.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
