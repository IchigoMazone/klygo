# `outputs.Box`

```python
Box(box, *, label, score=1.0, class_id=None, uid=None, track_id=None, metadata=None)
```

Stores one validated XYXY annotation. `uid` and `label` are read-only. `score`, `box`, `class_id`, `track_id`, and `metadata` are validated mutable properties. `translate()` and `scale()` mutate coordinates and return `None`.

```python
from klygo import outputs
box = outputs.Box([0, 0, 20, 10], label="car", score=0.9)
frame = outputs.Detection([box], source_id="sample", image_size=(100, 80))
frame[0].translate(5, 2)
print(frame[0].to_dict())
```

Invalid scores, coordinates, class IDs, UIDs, and tracking IDs raise `TypeError` or `ValueError` before mutation.
