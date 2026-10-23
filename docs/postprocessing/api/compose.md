# `postprocessing.compose`

```python
postprocessing.compose(*operations)
```

Build a reusable [`Pipeline`](Pipeline.md) from variadic callables or a single iterable of operations.

## Contract

Provides a flexible factory for [`Pipeline`](Pipeline.md):
- May be called with multiple positional callables: `post.compose(post.clip(), post.threshold(0.5), post.nms(0.45))`.
- May also be called with a single iterable (such as a `list`, `tuple`, or generator) of callables: `post.compose([post.clip(), post.threshold(0.5)])`.
- Nested `Pipeline` objects passed to `compose()` are automatically flattened into a single `Pipeline`.
- Calling `compose()` with no arguments returns an empty `Pipeline(())` that acts as an identity transformation.

## Parameters

- `*operations`: `Callable[[Any], Any] | Iterable[Callable[[Any], Any]]` — variadic callable transformations, or a single iterable of callable transformations.

## Returns

`Pipeline` — reusable postprocessing pipeline ready to be called on a `Detection`, `Detections`, or `Crops` object.

## Errors and edge cases

Raises `TypeError("Every postprocessing operation must be callable")` if any supplied stage is not callable, or if a single non-callable, non-iterable object is passed.

## Example

See [`compose.py`](../../../examples/postprocessing/compose.py) for a standalone runnable script.

## Complete executable example

```python
"""Compose variadic operations and iterable stage lists into a reusable Pipeline."""

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

# Variadic callables
pipeline = post.compose(post.clip(), post.threshold(0.5), post.top(1))
result = pipeline(frame)
assert result.labels == ["cat"]
assert result[0].box == [0.0, 0.0, 50.0, 50.0]

# Single iterable of operations produces an equivalent Pipeline
from_list = post.compose([post.clip(), post.threshold(0.5), post.top(1)])
assert from_list(frame).labels == ["cat"]
print(result.labels)
```
