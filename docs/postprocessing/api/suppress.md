# `postprocessing.suppress`

```python
postprocessing.suppress(result, iou=0.5, class_agnostic=False)
```

Apply score-ordered non-maximum suppression (NMS) to remove overlapping bounding boxes.

## Contract

Validates that `0.0 <= float(iou) <= 1.0` and performs greedy score-ordered suppression on each frame in `result` (`Detection`, `Detections`, or `Crops`):
- Sorts boxes in descending order of `box.score`.
- Repeatedly pops the highest-scoring remaining box `current`, appends it to `kept`, and filters the remaining candidates:
  - When `class_agnostic=False` (the default), only candidates with the same `class_id` as `current` and [`box_iou(current, candidate)`](box_iou.md) `> iou` are suppressed; candidates from other classes are always retained.
  - When `class_agnostic=True`, any candidate with `box_iou(current, candidate) > iou` is suppressed regardless of `class_id`.
- Candidates whose IoU with `current` is **less than or equal to** `iou` are retained.
- For a reusable pipeline stage, see [`postprocessing.nms(iou=0.5, class_agnostic=False)`](nms.md).

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to deduplicate.
- `iou`: `float`, default `0.5` — maximum allowed [`box_iou()`](box_iou.md) overlap in `[0.0, 1.0]` before a lower-scoring box is suppressed.
- `class_agnostic`: `bool`, default `False` — suppress overlapping boxes across different `class_id` values when `True`.

## Returns

`Detection | Detections | Crops` — new result containing the kept boxes sorted by descending score.

## Errors and edge cases

Raises `ValueError("iou must be between 0 and 1")` if `iou < 0` or `iou > 1`. Raises `TypeError("result must be Crops, Detection, or Detections")` for unsupported `result` types.

## Example

See [`suppress.py`](../../../examples/postprocessing/suppress.py) for a standalone runnable script.

## Complete executable example

```python
"""Run class-aware and class-agnostic non-maximum suppression on overlapping boxes."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(0, "cat", 0.9, [-5, 0, 50, 50]),
        Box(0, "cat", 0.8, [0, 0, 48, 48]),
        Box(1, "dog", 0.85, [0, 0, 49, 49]),
    ],
)

# Class-aware NMS suppresses the second cat (IoU > 0.5) but keeps the overlapping dog
class_aware = post.suppress(frame, iou=0.5, class_agnostic=False)
assert class_aware.labels == ["cat", "dog"]

# Class-agnostic NMS suppresses both lower-scoring overlapping boxes
agnostic = post.suppress(frame, iou=0.5, class_agnostic=True)
assert agnostic.labels == ["cat"]
print(class_aware.labels)
```
