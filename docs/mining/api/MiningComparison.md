# `mining.MiningComparison`

`MiningComparison` stores the complete before and after summaries plus their principal numeric changes.

```python
report = mining.compare(data, selected)
report.save("artifacts/mining/comparison.md")
```

See [`examples/mining/MiningComparison.py`](../../../examples/mining/MiningComparison.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
