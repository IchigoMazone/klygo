# `postprocessing.clip_boxes`

```python
postprocessing.clip_boxes(result, image_size=None)
```

Clamp bounding-box coordinates in `result` to the target `(width, height)` image bounds.

## Contract

Applies `box.clip(size)` to every `Box` in `result` (`Detection`, `Detections`, or `Crops`):
- When `image_size` is `None` (the default), uses each frame's own `frame.image_size` (`(width, height)` derived from `source_image`).
- When an explicit `image_size=(width, height)` tuple is provided, clamps `xmin` and `xmax` to `[0.0, float(width)]` and `ymin` and `ymax` to `[0.0, float(height)]`.
- Preserves box order and all frame metadata without mutating the input `result`. Note that boxes lying completely outside the image bounds become zero-area boxes on the boundary; chain [`remove_invalid_boxes()`](remove_invalid_boxes.md) afterward if zero-area boxes should be discarded.
- For a reusable pipeline stage, see [`postprocessing.clip(image_size=None)`](clip.md).

## Parameters

- `result`: `Detection | Detections | Crops` — detection output whose box coordinates will be clamped.
- `image_size`: `tuple[int, int] | None`, default `None` — optional `(width, height)` bounds; defaults to each frame's `image_size`.

## Returns

`Detection | Detections | Crops` — new result with every box clamped to `(width, height)`.

## Errors and edge cases

Raises `TypeError("result must be Crops, Detection, or Detections")` if `result` is not a supported detection container, or `TypeError` / `ValueError` if `image_size` is not a 2-element `(width, height)` sequence when supplied and `frame.image_size` is unavailable.

## Example

See [`clip_boxes.py`](../../../examples/postprocessing/clip_boxes.py) for a standalone runnable script.

## Complete executable example

```python
"""Clamp out-of-bounds box coordinates to the frame size or explicit bounds."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(0, "cat", 0.9, [-5, -10, 120, 95]),
        Box(1, "dog", 0.4, [60, 10, 90, 60]),
    ],
)

# Clip using frame.image_size (100, 80)
clipped = post.clip_boxes(frame)
assert clipped[0].box == [0.0, 0.0, 100.0, 80.0]
assert clipped[1].box == [60.0, 10.0, 90.0, 60.0]

# Clip to explicit custom bounds (40, 40)
custom_clipped = post.clip_boxes(frame, image_size=(40, 40))
assert custom_clipped[0].box == [0.0, 0.0, 40.0, 40.0]
print(clipped[0].box)
```
