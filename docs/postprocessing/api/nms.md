# `postprocessing.nms`

```python
postprocessing.nms(iou=0.5, class_agnostic=False)
```

Build a named [`Operation`](Operation.md) that applies score-ordered non-maximum suppression (NMS) to overlapping boxes.

## Contract

Returns `Operation("nms", lambda result: suppress(result, iou, class_agnostic))`, wrapping [`postprocessing.suppress()`](suppress.md) as a reusable pipeline stage for [`compose()`](compose.md), [`Pipeline`](Pipeline.md), and [`apply()`](apply.md):
- Sorts boxes by descending `box.score` and suppresses lower-scoring boxes whose [`box_iou()`](box_iou.md) with a kept box exceeds `iou`.
- Suppresses within each `class_id` independently when `class_agnostic=False` (default), or across all classes when `class_agnostic=True`.
- Preserves lazy evaluation when applied to `Detections` streams.

## Parameters

- `iou`: `float`, default `0.5` — maximum allowed IoU overlap in `[0.0, 1.0]` before suppression.
- `class_agnostic`: `bool`, default `False` — compare boxes across all classes regardless of `class_id` when `True`.

## Returns

`Operation` — frozen `Operation(nms)` callable.

## Errors and edge cases

Raises `ValueError("iou must be between 0 and 1")` when invoked if `iou < 0` or `iou > 1`, or `TypeError("result must be Crops, Detection, or Detections")` for unsupported `result` types.

## Example

See [`nms.py`](../../../examples/postprocessing/nms.py) for a standalone runnable script.

## Complete executable example

```python
"""Build a reusable NMS Operation and apply it standalone or inside a Pipeline."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(0, "cat", 0.9, [-5, 0, 50, 50]),
        Box(0, "cat", 0.8, [0, 0, 48, 48]),
        Box(1, "dog", 0.4, [60, 10, 90, 60]),
    ],
)

nms_op = post.nms(iou=0.5)
assert isinstance(nms_op, post.Operation) and nms_op.name == "nms"

deduped = nms_op(frame)
assert deduped.labels == ["cat", "dog"]

pipeline = post.compose(post.threshold(0.5), nms_op)
assert pipeline(frame).labels == ["cat"]
print(deduped.labels)
```
