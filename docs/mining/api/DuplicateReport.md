# `mining.DuplicateReport`

`DuplicateReport` stores duplicate index groups, exact groups, pair similarities, and cached perceptual hashes. `duplicate_images` reports how many images can be removed while keeping one representative per group.

```python
report = mining.find_duplicates(data)
print(report.duplicate_images)
```

See [`examples/mining/DuplicateReport.py`](../../../examples/mining/DuplicateReport.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
