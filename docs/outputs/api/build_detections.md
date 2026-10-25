# `outputs.build_detections`

```python
outputs.build_detections(images, predictions, *, metadata=None, names=None)
```

Builds one `Detection` per source image from backend-neutral prediction mappings. Each mapping contains equally sized `boxes`, `scores`, `labels` or `text_labels`, and optional `class_ids` sequences. Tensor-like values are converted through `detach`, `cpu`, and `tolist` when available.

Each returned frame owns stable integer box UIDs starting at zero, its source identity, image size, metadata, and a reconciled class registry. Labels are authoritative: an absent or conflicting preferred class ID is remapped to the ID already assigned to that label or the next free ID.

## Returns

`list[Detection]` with one materialized frame per image.

## Errors

Raises `ValueError` for mismatched batch or field lengths and invalid boxes. Raises `TypeError` when predictions are not mappings or their fields are not sequences.

## Example

See [`build_detections.py`](../../../examples/outputs/build_detections.py).

```python
from PIL import Image
from klygo import outputs

frames = outputs.build_detections(
    [Image.new("RGB", (80, 60))],
    [{"boxes": [[10, 10, 40, 35]], "scores": [0.92], "labels": ["car"]}],
    metadata={"backend": "custom"},
    names={0: "car"},
)
assert frames[0][0].uid == 0
assert frames[0][0].label == "car"
```

## Tests

See [`test_detect.py`](../../../test/outputs/test_detect.py).
