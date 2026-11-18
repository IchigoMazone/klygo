# `mining.cluster`

```python
cluster(data, *, features=("histogram", "hog", "quality", "boxes", "classes"), clusters=100, method="kmeans", seed=42, verbose=False)
```

Clusters complete images with standardized classical features and deterministic OpenCV K-Means. Supported features are color histogram, HOG, quality metrics, box statistics, and class presence.

```python
from klygo import mining
groups = mining.cluster(data, clusters=20, seed=42)
```

See [`examples/mining/cluster.py`](../../../examples/mining/cluster.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Multi-stage operations display sequential stage bars, never nested bars. Native K-Means displays a single stage because OpenCV does not expose iteration progress. Hash comparison progress counts completed outer rows; row costs decrease, so its ETA is approximate.
