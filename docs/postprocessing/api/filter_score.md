# `postprocessing.filter_score`

```python
postprocessing.filter_score(result, minimum)
```

Keep only bounding boxes whose confidence score is greater than or equal to `minimum`.

## Contract

Converts `minimum` to `float(minimum)` and delegates to [`select(result, where=lambda box: box.score >= threshold_value)`](select.md):
- Retains every `Box` with `box.score >= float(minimum)` in its original order.
- Leaves the input `result` (`Detection`, `Detections`, or `Crops`) unmutated and preserves all frame metadata.
- For a reusable [`Operation`](Operation.md) wrapper suitable for [`compose()`](compose.md) or [`Pipeline`](Pipeline.md), see [`postprocessing.threshold(minimum)`](threshold.md).

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to filter by confidence score.
- `minimum`: `float` — inclusive minimum confidence threshold (typically in `[0.0, 1.0]`).

## Returns

`Detection | Detections | Crops` — new result containing only boxes with `box.score >= float(minimum)`.

## Errors and edge cases

Raises `TypeError` or `ValueError` if `minimum` cannot be converted to `float`. Raises `TypeError("result must be Crops, Detection, or Detections")` if `result` is not a supported detection container.

## Example

See [`filter_score.py`](../../../examples/postprocessing/filter_score.py) for a standalone runnable script.

## Complete executable example

```python
"""Filter detection boxes directly by minimum confidence score."""

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

confident = post.filter_score(frame, 0.5)
assert confident.labels == ["cat", "cat"]
assert [b.score for b in confident] == [0.9, 0.8]

strict = post.filter_score(frame, 0.9)
assert len(strict) == 1 and strict[0].score == 0.9
print(confident.labels)
```
