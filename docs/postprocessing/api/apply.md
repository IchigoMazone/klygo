# `postprocessing.apply`

```python
postprocessing.apply(result, *operations)
```

Immediately apply one or more postprocessing operations in sequence to a detection result.

## Contract

Executes a one-shot transformation pipeline on `result` (`Detection`, `Detections`, or `Crops`):
- When `*operations` is empty, returns `result` unchanged without copying.
- When one or more operations (or a single iterable of operations) are supplied, builds a temporary [`Pipeline`](Pipeline.md) via [`compose(*operations)`](compose.md) and invokes it on `result`.
- Preserves input immutability: `result` is never modified in place, and lazy `Detections` streams are mapped lazily via `Detections.map(...)`.

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to transform.
- `*operations`: `Callable[[Any], Any] | Iterable[Callable[[Any], Any]]` — zero or more callable operations (or a single iterable of operations) to apply left-to-right.

## Returns

`Detection | Detections | Crops` — transformed result of the same container type as `result`.

## Errors and edge cases

Raises `TypeError("Every postprocessing operation must be callable")` if any supplied operation is not callable. Raises `TypeError("result must be Crops, Detection, or Detections")` when a built-in detection stage is applied to an unsupported `result` type.

## Example

See [`apply.py`](../../../examples/postprocessing/apply.py) for a standalone runnable script.

## Complete executable example

```python
"""Apply multiple postprocessing stages to a Detection in a single call."""

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

# Apply clip, confidence threshold, NMS, and top-1 selection immediately
cleaned = post.apply(
    frame,
    post.clip(),
    post.threshold(0.5),
    post.nms(0.5),
    post.top(1),
)
assert cleaned.labels == ["cat"]
assert cleaned[0].box == [0.0, 0.0, 50.0, 50.0]
assert post.apply(frame) is frame
print(cleaned.labels)
```
