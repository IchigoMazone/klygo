# `mining.MiningStats`

`MiningStats` stores aggregate image, annotation, class, co-occurrence, area, imbalance, and optional quality statistics.

```python
report = mining.stats(data)
report.show()
report.save("artifacts/mining/stats.json")
```

See [`examples/mining/MiningStats.py`](../../../examples/mining/MiningStats.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
