# `mining.boxes`

```python
boxes(data, *, min_boxes=0, max_boxes=None, min_area_ratio=0.0, limit=None, verbose=False)
```

Filters complete images by bounding-box count and minimum relative box area constraints, returning an independent `Detections`.

```python
from klygo import mining
dense = mining.boxes(data, min_boxes=2, min_area_ratio=0.002)
```

See [`examples/mining/boxes.py`](../../../examples/mining/boxes.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).
