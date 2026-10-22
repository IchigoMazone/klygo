# `postprocessing.threshold`

```python
postprocessing.threshold(minimum)
```

Build a named [`Operation`](Operation.md) that keeps boxes whose confidence score is at least `minimum`.

## Contract

Returns `Operation("threshold", lambda result: filter_score(result, minimum))`, which wraps [`postprocessing.filter_score()`](filter_score.md) for declarative use inside [`compose()`](compose.md), [`Pipeline`](Pipeline.md), or [`apply()`](apply.md):
- Calling the returned `Operation` on a `Detection`, `Detections`, or `Crops` retains all boxes with `box.score >= float(minimum)` without mutating the input.
- When included in a [`Pipeline`](Pipeline.md) and applied to a `Detections` stream, the threshold operation is attached lazily via `Detections.map(...)`.

## Parameters

- `minimum`: `float` — inclusive minimum confidence score required for a box to be retained.

## Returns

`Operation` — frozen `Operation(threshold)` callable.

## Errors and edge cases

Raises `TypeError` or `ValueError` when invoked if `minimum` cannot be converted to `float`, or `TypeError("result must be Crops, Detection, or Detections")` if the target `result` is not a supported detection container.

## Example

See [`threshold.py`](../../../examples/postprocessing/threshold.py) for a standalone runnable script.

## Complete executable example

```python
"""Create a reusable score-threshold Operation and use it standalone or in a Pipeline."""

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

step = post.threshold(0.5)
assert isinstance(step, post.Operation) and step.name == "threshold"

filtered = step(frame)
assert filtered.labels == ["cat", "cat"]

pipeline = post.compose(step, post.top(1))
assert pipeline(frame).labels == ["cat"]
print(filtered.labels)
```
