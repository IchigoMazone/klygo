# `mining.cluster`

```python
cluster(data, *, features=("histogram", "hog", "quality", "boxes", "classes"), clusters=100, method="kmeans", seed=42)
```

Clusters complete images with standardized classical features and deterministic OpenCV K-Means. Supported features are color histogram, HOG, quality metrics, box statistics, and class presence.

```python
from klygo import mining
groups = mining.cluster(data, clusters=20, seed=42)
```

See [`examples/mining/cluster.py`](../../../examples/mining/cluster.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
