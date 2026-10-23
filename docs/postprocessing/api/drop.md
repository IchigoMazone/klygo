# `postprocessing.drop`

```python
postprocessing.drop(result, *, ids=None, class_ids=None, uids=None, labels=None, where=None)
```

Remove bounding boxes that match all supplied selector criteria.

## Contract

Acts as the exact complement of [`postprocessing.select()`](select.md). Evaluates each `Box` in `result` (`Detection`, `Detections`, or `Crops`) against all provided keyword selectors (`ids`, `class_ids`, `uids`, `labels`, `where`):
- A box is dropped if and only if it matches **every** non-`None` selector provided.
- Note: if no selectors are supplied (all keyword arguments are `None`), every box matches the empty conjunction and all boxes are dropped.
- Returns a new `Detection`, `Crops`, or lazily mapped `Detections` without modifying the original `result`.

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to filter.
- `ids`: `Iterable[int] | None`, default `None` — keyword-only; `box.id` values to match for removal.
- `class_ids`: `Iterable[int] | None`, default `None` — keyword-only; `box.class_id` values to match for removal.
- `uids`: `Iterable[str] | None`, default `None` — keyword-only; `box.uid` strings to match for removal.
- `labels`: `Iterable[str] | None`, default `None` — keyword-only; `box.label` names to match for removal.
- `where`: `Callable[[Box], bool] | None`, default `None` — keyword-only; predicate returning `True` for boxes to drop.

## Returns

`Detection | Detections | Crops` — new result with matching boxes removed.

## Errors and edge cases

Raises `TypeError("result must be Crops, Detection, or Detections")` if `result` is not a supported detection container, or `TypeError` if any supplied selector iterable is not iterable or `where` is not callable.

## Example

See [`drop.py`](../../../examples/postprocessing/drop.py) for a standalone runnable script.

## Complete executable example

```python
"""Drop unwanted labels or low-confidence detections from a Detection frame."""

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

# Drop all boxes labeled "dog"
without_dogs = post.drop(frame, labels=["dog"])
assert without_dogs.labels == ["cat", "cat"]

# Drop only "cat" boxes whose score is below 0.85
high_cats_and_dogs = post.drop(frame, labels=["cat"], where=lambda b: b.score < 0.85)
assert high_cats_and_dogs.labels == ["cat", "dog"]
print(without_dogs.labels)
```
