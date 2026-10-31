# `mining.MiningScores`

`MiningScores` stores the final score, component values, weights, and source identity for every image.

```python
report = mining.score(data)
report.save("artifacts/mining/scores.csv")
```

See [`examples/mining/MiningScores.py`](../../../examples/mining/MiningScores.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
