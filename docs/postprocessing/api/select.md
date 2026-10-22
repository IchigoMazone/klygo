# `postprocessing.select`

```python
postprocessing.select(result, *, ids=None, class_ids=None, uids=None, labels=None, where=None)
```

Keep only bounding boxes that satisfy every supplied selector criterion.

## Contract

Evaluates each `Box` in `result` (`Detection`, `Detections`, or `Crops`) against all provided keyword selectors using logical AND:
- When a selector (`ids`, `class_ids`, `uids`, `labels`) is `None`, that attribute is not restricted.
- When an iterable is provided for a selector, converts it into a `set` once and requires the box's corresponding attribute (`box.id`, `box.class_id`, `box.uid`, `box.label`) to be a member of that set.
- When `where` is provided, requires `where(box)` to return truthy.
- Returns a new `Detection` (preserving `source_image`, `speed`, `image_frame_index`, `config`, `names`, `timestamp`, `source_path`, and `url`), a new `Crops`, or a lazily mapped `Detections` without mutating `result`.

## Parameters

- `result`: `Detection | Detections | Crops` — detection output to filter.
- `ids`: `Iterable[int] | None`, default `None` — keyword-only; allowed `box.id` values.
- `class_ids`: `Iterable[int] | None`, default `None` — keyword-only; allowed `box.class_id` integer indices.
- `uids`: `Iterable[str] | None`, default `None` — keyword-only; allowed `box.uid` unique identifiers.
- `labels`: `Iterable[str] | None`, default `None` — keyword-only; allowed `box.label` class names.
- `where`: `Callable[[Box], bool] | None`, default `None` — keyword-only; custom predicate evaluated on each `Box`.

## Returns

`Detection | Detections | Crops` — new result containing only matching boxes.

## Errors and edge cases

Raises `TypeError("result must be Crops, Detection, or Detections")` if `result` is not a supported detection container, or `TypeError` if any selector iterable is not iterable or `where` is not callable. Passing an empty iterable (such as `labels=[]`) matches zero boxes.

## Example

See [`select.py`](../../../examples/postprocessing/select.py) for a standalone runnable script.

## Complete executable example

```python
"""Select boxes by label, class_id, and custom geometric predicate."""

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

cats = post.select(frame, labels=["cat"])
assert cats.labels == ["cat", "cat"]

high_conf_cats = post.select(frame, class_ids=[0], where=lambda box: box.score >= 0.85)
assert len(high_conf_cats) == 1 and high_conf_cats[0].score == 0.9
print(cats.labels)
```
