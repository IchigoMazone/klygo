# `postprocessing.top`

```python
postprocessing.top(count)
```

Build a named [`Operation`](Operation.md) that retains at most the highest-scoring `count` boxes per frame.

## Contract

Returns `Operation("top", lambda result: top_boxes(result, count))`, wrapping [`postprocessing.top_boxes()`](top_boxes.md) as a composable pipeline stage for [`compose()`](compose.md), [`Pipeline`](Pipeline.md), and [`apply()`](apply.md):
- Sorts each frame's boxes by descending `box.score` and keeps at most `int(count)` boxes.
- Typically placed at the end of a postprocessing pipeline (after [`clip()`](clip.md), [`remove_invalid()`](remove_invalid.md), [`threshold()`](threshold.md), and [`nms()`](nms.md)) to cap the maximum number of detections emitted per frame.

## Parameters

- `count`: `int` — non-negative maximum number of highest-scoring boxes to retain per frame.

## Returns

`Operation` — frozen `Operation(top)` callable.

## Errors and edge cases

Raises `ValueError("count must be greater than or equal to 0")` when invoked if `int(count) < 0`, or `TypeError("result must be Crops, Detection, or Detections")` if the target `result` is not a supported detection container.

## Example

See [`top.py`](../../../examples/postprocessing/top.py) for a standalone runnable script.

## Complete executable example

```python
"""Build a top-K Operation and combine it with threshold and NMS stages."""

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

top_two = post.top(2)
assert isinstance(top_two, post.Operation) and top_two.name == "top"

result = top_two(frame)
assert result.labels == ["cat", "cat"]
assert [b.score for b in result] == [0.9, 0.8]

pipeline = post.compose(post.clip(), post.nms(0.5), post.top(1))
assert pipeline(frame).labels == ["cat"]
print(result.labels)
```
