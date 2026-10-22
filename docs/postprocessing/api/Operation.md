# `postprocessing.Operation`

```python
postprocessing.Operation(name, function)
```

Represent an immutable, named callable transformation applied to a detection result.

## Contract

`Operation` is a frozen `@dataclass(frozen=True)` with two attributes (`name` and `function`). Calling `operation(result)` invokes `self.function(result)` and returns its output directly, while `repr(operation)` formats as `Operation(<name>)` for readable inspection inside a [`Pipeline`](Pipeline.md).

All built-in pipeline stage builders in `klygo.postprocessing` ([`threshold()`](threshold.md), [`clip()`](clip.md), [`remove_invalid()`](remove_invalid.md), [`nms()`](nms.md), [`top()`](top.md), and [`custom()`](custom.md)) return `Operation` instances. You can also instantiate `Operation(name, function)` directly whenever you want explicit control over the operation name and underlying callable.

## Parameters

- `name`: `str` — human-readable identifier displayed in `repr(operation)` and [`Pipeline`](Pipeline.md) summaries.
- `function`: `Callable[[Any], Any]` — transformation function accepting a `Detection`, `Detections`, or `Crops` result and returning the transformed result.

## Returns

`Operation` — frozen callable dataclass instance.

## Errors and edge cases

Attempting to mutate `.name` or `.function` after construction raises `dataclasses.FrozenInstanceError`. If `function` is not callable, a `TypeError` is raised when `operation(result)` is invoked or when the `Operation` is validated inside [`custom()`](custom.md).

## Example

See [`Operation.py`](../../../examples/postprocessing/Operation.py) for a standalone runnable script.

## Complete executable example

```python
"""Construct and invoke a named postprocessing Operation directly and in a Pipeline."""

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

keep_cats = post.Operation("keep_cats", lambda result: post.select(result, labels=["cat"]))
assert keep_cats.name == "keep_cats"
assert repr(keep_cats) == "Operation(keep_cats)"

filtered = keep_cats(frame)
assert filtered.labels == ["cat", "cat"]
assert len(frame) == 3  # original Detection is never mutated
print(filtered.labels)
```
