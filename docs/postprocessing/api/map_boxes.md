# `postprocessing.map_boxes`

```python
postprocessing.map_boxes(result, function)
```

Transform each `Box` in a detection result with a callable, dropping any box for which the callable returns `None`.

## Contract

Validates that `function` is callable and applies it to every `Box` in `result` (`Detection`, `Detections`, or `Crops`):
- If `function(box)` returns a `Box` instance (such as `box.with_label(...)` or `box.clip(...)`), that transformed `Box` is appended to the output frame.
- If `function(box)` returns `None`, that box is omitted from the output frame, allowing simultaneous transformation and filtering in a single pass.
- Preserves all frame-level metadata (`source_image`, `speed`, `image_frame_index`, `config`, `names`, `timestamp`, `source_path`, `url`) and maps lazily over `Detections` streams.

## Parameters

- `result`: `Detection | Detections | Crops` — detection output whose boxes will be mapped.
- `function`: `Callable[[Box], Box | None]` — function called once per `Box`, returning a replacement `Box` or `None`.

## Returns

`Detection | Detections | Crops` — new result containing the mapped (and non-`None`) boxes.

## Errors and edge cases

Raises `TypeError("function must be callable")` if `function` is not callable. Raises `TypeError("map_boxes function must return Box or None")` during execution if `function(box)` returns any value other than a `Box` instance or `None`. Raises `TypeError("result must be Crops, Detection, or Detections")` for unsupported `result` types.

## Example

See [`map_boxes.py`](../../../examples/postprocessing/map_boxes.py) for a standalone runnable script.

## Complete executable example

```python
"""Relabel and conditionally drop boxes in a single map_boxes pass."""

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

# Relabel every box to "animal"
all_animals = post.map_boxes(frame, lambda box: box.with_label("animal"))
assert all_animals.labels == ["animal", "animal", "animal"]

# Keep only boxes with score >= 0.5 and clip them to (100, 80)
clipped_confident = post.map_boxes(
    frame,
    lambda box: box.clip((100, 80)) if box.score >= 0.5 else None,
)
assert len(clipped_confident) == 2 and clipped_confident[0].box == [0.0, 0.0, 50.0, 50.0]
print(all_animals.labels)
```
