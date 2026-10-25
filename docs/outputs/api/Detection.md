# `outputs.Detection`

```python
Detection(boxes=None, *, source_id=None, frame_index=0, source_path=None, image_size=None, source=None, metadata=None, names=None)
```

Represents one source frame. It provides positional access, stable UID lookup, atomic box CRUD, frame-level `filter`, `map`, `drop`, and `relabel`, three display caption modes, and annotation serialization.

`names` contains only active classes and maps each active `class_id` to its label. Adding, replacing, relabeling, or deleting boxes reconciles this mapping before the mutation commits.

```python
from klygo import outputs
frame = outputs.Detection([], source_id="sample", frame_index=0, image_size=(100, 80))
frame.add_box([5, 5, 30, 25], label="car", score=0.8)
frame.show(mode="uid")
```

`show()` supports `default`, `index`, and `uid`. It does not change annotations.
