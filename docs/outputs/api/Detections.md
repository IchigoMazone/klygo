# `outputs.Detections`

```python
Detections(frames=(), *, metadata=None, names=None)
```

Owns an ordered materialized frame collection. It provides collision-safe CRUD, stable frame lookup, net change properties, mutation history, class-wide `drop` and `relabel`, backend-neutral `repredict`, JSON serialization, and annotation-only YOLO export.

The collection owns the shared class registry. Frame CRUD and direct mutations on an owned `Detection` automatically reconcile every active label so the same label has one `class_id` across all frames. Removed labels disappear from `names`.

Use `preview_add(frame)` before `append(frame)` to inspect identity collisions, new labels, and exact class-ID remapping without mutating either object.

```python
preview = results.preview_add(frame)
if preview["can_add"]:
    results.append(frame)
```

```python
from klygo import models, outputs
model = models.load("yolov8n.pt")
results = model.predict("images/", stream=False)
results.drop(["unknown"])
results.repredict(model, index=[0, 2], threshold=0.5)
results.export("artifacts/predictions", format="yolo")
```

`merge()` raises on any duplicate `(source_id, frame_index)` identity. Batch operations validate before committing and return `None`.
