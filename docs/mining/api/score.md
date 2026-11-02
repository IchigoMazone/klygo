# `mining.score`

```python
score(data, *, weights=None)
```

Returns one normalized score per image. `weights` may contain `quality`, `rarity`, `coverage`, and `boxes`; omitted weights use a practical default blend.

```python
from klygo import mining
scores = mining.score(data, weights={"quality": 0.6, "rarity": 0.4})
```

See [`examples/mining/score.py`](../../../examples/mining/score.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
