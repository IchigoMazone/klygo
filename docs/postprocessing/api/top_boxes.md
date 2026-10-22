# `postprocessing.top_boxes`

```python
postprocessing.top_boxes(result, count)
```

Keep at most the `count` highest-scoring bounding boxes per frame, ordered by descending confidence score.

## Contract

Converts `count` to `int(count)`, validates `count >= 0`, and transforms each frame in `result` (`Detection`, `Detections`, or `Crops`):
- Sorts `frame.objects` by `box.score` in descending order (`reverse=True`) using Python's stable Timsort (ties preserve relative input order).
- Slices the first `count` boxes (`[:limit]`) and constructs a new `Detection` (or `Crops` / lazily mapped `Detections`) preserving all frame metadata.
- If a frame contains fewer than `count` boxes, all boxes in that frame are returned sorted by descending score. If `count == 0`, returns an empty box list per frame.
- For a reusable pipeline stage, see [`postprocessing.top(count)`](top.md).

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to truncate per frame.
- `count`: `int` — non-negative maximum number of boxes to retain per frame.

## Returns

`Detection | Detections | Crops` — new result with at most `count` highest-scoring boxes per frame.

## Errors and edge cases

Raises `ValueError("count must be greater than or equal to 0")` when `int(count) < 0`. Raises `TypeError` or `ValueError` if `count` cannot be converted to `int`, and `TypeError("result must be Crops, Detection, or Detections")` for unsupported `result` types.

## Example

See [`top_boxes.py`](../../../examples/postprocessing/top_boxes.py) for a standalone runnable script.

## Complete executable example

```python
"""Retain the highest-scoring K boxes from a Detection frame."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(1, "dog", 0.4, [60, 10, 90, 60]),
        Box(0, "cat", 0.9, [-5, 0, 50, 50]),
        Box(0, "cat", 0.8, [0, 0, 48, 48]),
    ],
)

top_two = post.top_boxes(frame, 2)
assert top_two.labels == ["cat", "cat"]
assert [b.score for b in top_two] == [0.9, 0.8]

empty = post.top_boxes(frame, 0)
assert len(empty) == 0
print(top_two.labels)
```
