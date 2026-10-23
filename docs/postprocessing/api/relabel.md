# `postprocessing.relabel`

```python
postprocessing.relabel(result, label, *, ids=None, class_ids=None, uids=None, labels=None, where=None)
```

Relabel matching bounding boxes using a string, ID-to-label mapping, or callable while keeping all non-matching boxes unchanged.

## Contract

Evaluates every `Box` in `result` (`Detection`, `Detections`, or `Crops`) against the optional keyword selectors (`ids`, `class_ids`, `uids`, `labels`, `where`):
- Boxes that do **not** match all supplied selectors are retained unchanged in their original order.
- For each matching `Box`, computes `new_label`:
  - If `label` is a `Mapping` (e.g., `dict[int, str]`), looks up `label.get(box.id, box.label)`.
  - If `label` is a callable, calls `label(box)`.
  - Otherwise uses `label` as a constant string.
- Produces a new box via `box.with_label(new_label)` without mutating the input `Box` or `result`.

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to relabel.
- `label`: `str | Mapping[int, str] | Callable[[Box], str]` — replacement label string, mapping from `box.id` to label string, or function `Box -> str`.
- `ids`: `Iterable[int] | None`, default `None` — keyword-only; restrict relabeling to these `box.id` values.
- `class_ids`: `Iterable[int] | None`, default `None` — keyword-only; restrict relabeling to these `box.class_id` values.
- `uids`: `Iterable[str] | None`, default `None` — keyword-only; restrict relabeling to these `box.uid` values.
- `labels`: `Iterable[str] | None`, default `None` — keyword-only; restrict relabeling to these `box.label` names.
- `where`: `Callable[[Box], bool] | None`, default `None` — keyword-only; restrict relabeling to boxes satisfying `where(box)`.

## Returns

`Detection | Detections | Crops` — new result with matching boxes relabeled and all other boxes preserved.

## Errors and edge cases

Raises `TypeError("result must be Crops, Detection, or Detections")` if `result` is not a supported detection container.

## Example

See [`relabel.py`](../../../examples/postprocessing/relabel.py) for a standalone runnable script.

## Complete executable example

```python
"""Relabel selected boxes by constant string, mapping, or callable."""

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

# Relabel only "cat" boxes to "feline"
felines = post.relabel(frame, "feline", labels=["cat"])
assert felines.labels == ["feline", "feline", "dog"]

# Relabel using a callable formatting the class name and score
formatted = post.relabel(frame, lambda b: f"{b.label}:{b.score:.1f}")
assert formatted.labels == ["cat:0.9", "cat:0.8", "dog:0.4"]
print(felines.labels)
```
