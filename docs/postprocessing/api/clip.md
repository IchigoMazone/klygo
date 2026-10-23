# `postprocessing.clip`

```python
postprocessing.clip(image_size=None)
```

Build a named [`Operation`](Operation.md) that clamps bounding-box coordinates to the image bounds.

## Contract

Returns `Operation("clip", lambda result: clip_boxes(result, image_size))`, wrapping [`postprocessing.clip_boxes()`](clip_boxes.md) as a composable pipeline stage for [`compose()`](compose.md), [`Pipeline`](Pipeline.md), and [`apply()`](apply.md):
- When `image_size=None` (the default), each frame's boxes are clamped to that frame's own `frame.image_size` (`(width, height)`).
- When `image_size=(width, height)` is specified, clamps `xmin`/`xmax` to `[0.0, float(width)]` and `ymin`/`ymax` to `[0.0, float(height)]`.
- Never mutates the input `Detection`, `Detections`, or `Crops`.

## Parameters

- `image_size`: `tuple[int, int] | None`, default `None` — explicit `(width, height)` bounds, or `None` to use each frame's `image_size`.

## Returns

`Operation` — frozen `Operation(clip)` callable.

## Errors and edge cases

Raises `TypeError("result must be Crops, Detection, or Detections")` when invoked on an unsupported result object, or `TypeError` / `ValueError` if `image_size` is invalid when called.

## Example

See [`clip.py`](../../../examples/postprocessing/clip.py) for a standalone runnable script.

## Complete executable example

```python
"""Build a coordinate-clipping Operation and combine it with invalid-box removal."""

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

clip_op = post.clip()
assert isinstance(clip_op, post.Operation) and clip_op.name == "clip"

clipped = clip_op(frame)
assert clipped[0].box == [0.0, 0.0, 50.0, 50.0]

# Clip to explicit (40, 40) bounds inside a composed pipeline
custom_pipe = post.compose(post.clip((40, 40)), post.remove_invalid(min_area=1))
assert len(custom_pipe(frame)) == 2
print(clipped[0].box)
```
