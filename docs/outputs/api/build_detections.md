# `outputs.build_detections`

```python
outputs.build_detections(images, predictions, *, metadata=None, names=None)
```

Build one `Detection` per image from backend-neutral prediction mappings.
Every prediction supplies equally sized `boxes`, `scores`, and `labels`
sequences. An optional `class_ids` sequence sets `Box.class_id`; `Box.id`
remains the frame-local box identity. `names` supplies the model vocabulary,
and labels missing from it are appended to the result registry.

The function returns `list[Detection]`. `Detector.predict()` remains
responsible for wrapping batch results in `Detections` and attaching timing,
source, frame, and streaming information. Each frame receives an independent
snapshot of `metadata`, including nested lists and dictionaries.

```python
from PIL import Image

from klygo import outputs

image = Image.new("RGB", (80, 60), "white")
frames = outputs.build_detections(
    [image],
    [{
        "boxes": [[10, 10, 40, 35]],
        "scores": [0.92],
        "labels": ["item"],
        "class_ids": [3],
    }],
    names={3: "item"},
)

print(frames[0][0].id)        # 0: box identity
print(frames[0][0].class_id)  # 3: class identity
print(frames[0].names)        # {3: "item"}
```

See the [runnable example](../../../examples/outputs/build_detections.py) and
the [`klygo.outputs` index](../README.md).
