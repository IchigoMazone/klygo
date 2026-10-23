# `postprocessing.box_iou`

```python
postprocessing.box_iou(first, second)
```

Compute the intersection-over-union (IoU) ratio in `[0.0, 1.0]` for two XYXY bounding boxes.

## Contract

Accepts either `Box` instances or 4-element coordinate sequences (`[xmin, ymin, xmax, ymax]`) for `first` and `second`:
- Extracts the first four coordinates `ax1, ay1, ax2, ay2` and `bx1, by1, bx2, by2` as `float`.
- Computes intersection width `max(0.0, min(ax2, bx2) - max(ax1, bx1))` and height `max(0.0, min(ay2, by2) - max(ay1, by1))`.
- Computes individual areas clamping negative extents to `0.0`, and `union = area_a + area_b - intersection`.
- Returns `intersection / union` when `union > 0`, or `0.0` when both boxes are degenerate or have zero union area.

## Parameters

- `first`: `Box | Sequence[float]` — first bounding box or `[xmin, ymin, xmax, ymax]` coordinate sequence.
- `second`: `Box | Sequence[float]` — second bounding box or `[xmin, ymin, xmax, ymax]` coordinate sequence.

## Returns

`float` — intersection-over-union ratio in `[0.0, 1.0]`.

## Errors and edge cases

Raises `TypeError` or `IndexError` / `ValueError` if `first` or `second` is neither a `Box` nor a sequence of at least 4 numeric coordinates. Touching edges or corners with zero overlap area return `0.0`; degenerate zero-area boxes return `0.0` without dividing by zero.

## Example

See [`box_iou.py`](../../../examples/postprocessing/box_iou.py) for a standalone runnable script.

## Complete executable example

```python
"""Compute IoU between Box instances and raw XYXY coordinate sequences."""

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

# IoU between two Box objects
overlap = round(post.box_iou(frame[0], frame[1]), 3)
assert overlap == 0.834

# Identical and disjoint coordinate sequences
assert post.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
assert post.box_iou(frame[0], frame[2]) == 0.0
print(overlap)
```
