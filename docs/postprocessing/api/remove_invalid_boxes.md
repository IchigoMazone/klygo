# `postprocessing.remove_invalid_boxes`

```python
postprocessing.remove_invalid_boxes(result, min_area=0)
```

Remove boxes with non-finite coordinates, non-positive width or height, or area smaller than `min_area`.

## Contract

Filters `result` (`Detection`, `Detections`, or `Crops`) via [`select()`](select.md), retaining only boxes that satisfy all five validity checks:
1. `len(box.box) == 4`
2. Every coordinate (`xmin, ymin, xmax, ymax`) is finite (`math.isfinite(value)` is `True`, rejecting `NaN` and `±inf`).
3. Strictly positive width: `box.xmax > box.xmin`.
4. Strictly positive height: `box.ymax > box.ymin`.
5. Minimum area met: `box.area >= float(min_area)`.

Even when `min_area=0` (the default), zero-width (`xmax == xmin`), zero-height (`ymax == ymin`), and inverted (`xmax < xmin` or `ymax < ymin`) boxes are removed because checks 3 and 4 require strict inequality. For a reusable pipeline stage, see [`postprocessing.remove_invalid(min_area=0)`](remove_invalid.md).

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to sanitize.
- `min_area`: `float`, default `0` — minimum inclusive bounding-box area `(xmax - xmin) * (ymax - ymin)` required to keep a valid box.

## Returns

`Detection | Detections | Crops` — new result containing only finite, positively sized boxes with `area >= float(min_area)`.

## Errors and edge cases

Raises `TypeError` or `ValueError` if `min_area` cannot be converted to `float`. Raises `TypeError("result must be Crops, Detection, or Detections")` if `result` is not a supported detection container.

## Example

See [`remove_invalid_boxes.py`](../../../examples/postprocessing/remove_invalid_boxes.py) for a standalone runnable script.

## Complete executable example

```python
"""Filter out inverted, degenerate, NaN, and undersized bounding boxes."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(0, "cat", 0.9, [0, 0, 50, 50]),          # area = 2500 (valid)
        Box(0, "tiny", 0.8, [0, 0, 5, 5]),           # area = 25 (small)
        Box(1, "inverted", 0.7, [50, 50, 10, 10]),   # xmax < xmin (invalid)
        Box(1, "nan_box", 0.6, [0, 0, float("nan"), 20]),
    ],
)

valid_default = post.remove_invalid_boxes(frame)
assert valid_default.labels == ["cat", "tiny"]

valid_min_100 = post.remove_invalid_boxes(frame, min_area=100)
assert valid_min_100.labels == ["cat"]
print(len(valid_min_100), valid_min_100.labels)
```
