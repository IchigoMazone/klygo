# `mining.classes`

```python
classes(data, classes, *, match="any", limit=None, verbose=False)
```

Filters complete images by class label inclusion or exclusion rules and returns an independent `Detections`. `match` accepts `any`, `all`, `only`, or `exclude`. Unknown class names raise `ValueError`.

```python
from klygo import mining
cars = mining.classes(data, ["car"], match="any")
```

See [`examples/mining/classes.py`](../../../examples/mining/classes.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
