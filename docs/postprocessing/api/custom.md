# `postprocessing.custom`

```python
postprocessing.custom(function, name=None)
```

Wrap an ordinary Python callable into a named [`Operation`](Operation.md) for use in postprocessing pipelines.

## Contract

Validates that `function` is callable and wraps it in a frozen [`Operation(name, function)`](Operation.md):
- If `name` is explicitly provided and truthy, uses `name` as the operation's identifier.
- Otherwise falls back to `getattr(function, "__name__", "custom")`.
- The returned `Operation` can be called directly on a `Detection`, `Detections`, or `Crops` result, or combined with built-in stages inside [`compose()`](compose.md), [`Pipeline`](Pipeline.md), and [`apply()`](apply.md). When a [`Pipeline`](Pipeline.md) executes on a `Detections` collection, `function` is invoked per frame via `Detections.map(...)`.

## Parameters

- `function`: `Callable[[Any], Any]` — user-defined transformation function accepting a detection result and returning the transformed result.
- `name`: `str | None`, default `None` — optional display name for the operation; defaults to `function.__name__` or `'custom'`.

## Returns

`Operation` — named operation wrapping `function`.

## Errors and edge cases

Raises `TypeError("function must be callable")` immediately if `function` is not callable.

## Example

See [`custom.py`](../../../examples/postprocessing/custom.py) for a standalone runnable script.

## Complete executable example

```python
"""Wrap a custom filtering function as a named Operation and compose it in a Pipeline."""

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

keep_cats = post.custom(lambda result: post.select(result, labels=["cat"]), name="cats")
assert repr(keep_cats) == "Operation(cats)"
assert keep_cats(frame).labels == ["cat", "cat"]

# Combine custom operation with built-in pipeline stages
pipeline = post.compose(keep_cats, post.nms(0.5))
assert pipeline(frame).labels == ["cat"]
print(keep_cats(frame).labels)
```
