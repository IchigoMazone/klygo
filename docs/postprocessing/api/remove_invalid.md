# `postprocessing.remove_invalid`

```python
postprocessing.remove_invalid(min_area=0)
```

Build a named [`Operation`](Operation.md) that removes non-finite, inverted, zero-size, or undersized bounding boxes.

## Contract

Returns `Operation("remove_invalid", lambda result: remove_invalid_boxes(result, min_area))`, wrapping [`postprocessing.remove_invalid_boxes()`](remove_invalid_boxes.md) for declarative use inside [`compose()`](compose.md), [`Pipeline`](Pipeline.md), and [`apply()`](apply.md):
- Keeps only boxes with 4 finite coordinates, `xmax > xmin`, `ymax > ymin`, and `area >= float(min_area)`.
- Commonly placed right after [`post.clip()`](clip.md) in a pipeline so boxes that were completely outside the image frame (and clamped to zero width or height at the image border) are cleanly discarded.

## Parameters

- `min_area`: `float`, default `0` — minimum inclusive area `(xmax - xmin) * (ymax - ymin)` required for a valid box to be retained.

## Returns

`Operation` — frozen `Operation(remove_invalid)` callable.

## Errors and edge cases

Raises `TypeError` or `ValueError` when invoked if `min_area` cannot be converted to `float`, or `TypeError("result must be Crops, Detection, or Detections")` if the target `result` is not a supported detection container.

## Example

See [`remove_invalid.py`](../../../examples/postprocessing/remove_invalid.py) for a standalone runnable script.

## Complete executable example

```python
"""Create a remove_invalid Operation to sanitize boxes in a Detection or Pipeline."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(0, "cat", 0.9, [-5, 0, 50, 50]),
        Box(0, "tiny", 0.8, [0, 0, 5, 5]),
        Box(1, "dog", 0.4, [60, 10, 90, 60]),
    ],
)

sanitize = post.remove_invalid(min_area=100)
assert isinstance(sanitize, post.Operation) and sanitize.name == "remove_invalid"

cleaned = sanitize(frame)
assert len(cleaned) == 2 and cleaned.labels == ["cat", "dog"]

pipeline = post.compose(post.clip(), sanitize)
assert [b.box for b in pipeline(frame)] == [[0.0, 0.0, 50.0, 50.0], [60.0, 10.0, 90.0, 60.0]]
print(len(cleaned))
```
