# `mining.ClusterResult`

`ClusterResult` stores one cluster label per image, index groups, representative indices, feature names, the seed, and the standardized feature matrix reused by diversity selection.

```python
result = mining.cluster(data, clusters=20)
result.show()
```

See [`examples/mining/ClusterResult.py`](../../../examples/mining/ClusterResult.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
