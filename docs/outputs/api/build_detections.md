# `outputs.build_detections`

```python
outputs.build_detections(
    images,
    predictions,
    *,
    metadata=None,
    names=None,
)
```

Build one `Detection` per image from normalized backend prediction mappings.

## Contract

Converts a sequence of source images and a matching sequence of backend-neutral prediction mappings into a `list[Detection]`. Each prediction mapping supplies equally sized `boxes`, `scores`, and `labels` (or `text_labels`) sequences. Tensor-like objects exposing `.detach()`, `.cpu()`, or `.tolist()` are automatically converted to plain Python lists, coordinates are rounded to 2 decimal places, and confidence scores are rounded to 3 decimal places.

Within each frame, `Box.id` is assigned sequentially (`0, 1, ...`) to identify the box inside that frame, while `Box.class_id` tracks the class vocabulary index. When a prediction includes `class_ids`, those integers become `Box.class_id` and are verified against `names`. When `class_ids` is omitted, each label is looked up in `names` or appended with the next available integer ID in first-seen order. Every returned `Detection` receives an independent deep copy of `metadata` as its `config` dictionary and shares the finalized `_all_names` vocabulary registry across the batch.

## Parameters

- `images`: `Sequence[Any]` — source images (such as `PIL.Image.Image` or `media.LazyImage` instances), one per frame.
- `predictions`: `Sequence[Mapping[str, Any]]` — normalized prediction mappings matching `images` by position. Each mapping may contain:
  - `"boxes"`: sequence of 4-number `[xmin, ymin, xmax, ymax]` coordinates.
  - `"scores"`: sequence of numeric confidence scores.
  - `"labels"` or `"text_labels"`: sequence of class label strings (`"text_labels"` takes precedence when present).
  - `"class_ids"` (optional): sequence of integer class IDs matching `"boxes"`.
- `metadata`: `Mapping[str, Any] | None`, default `None` — optional frame configuration metadata deep-copied into each `Detection.config`; keyword-only.
- `names`: `Mapping[int, Any] | Sequence[Any] | None`, default `None` — initial class vocabulary mapping or sequence of class names; keyword-only.

## Returns

`list[Detection]` — one `Detection` instance per input image, populated with `Box` objects, frame metadata, and synchronized class registries.

## Errors and edge cases

- Raises `ValueError` if `images` and `predictions` have different lengths.
- Raises `TypeError` if `metadata` is not a `Mapping` or `None`, if any element of `predictions` is not a `Mapping`, or if `boxes`, `scores`, `labels`, or `class_ids` cannot be converted to a sequence.
- Raises `ValueError` if `boxes`, `scores`, and `labels` within a prediction do not have the exact same length, if `class_ids` is provided with a different length than `boxes`, or if any box does not contain exactly 4 coordinates.
- Raises `ValueError` if an explicit `class_id` maps to two conflicting label names within the same batch or `names` registry.

## AI usage guidance

Prefer `from klygo import outputs` followed by `outputs.build_detections(...)` when adapting custom model outputs into Klygo result containers. Keep `Box.id` reserved for frame-local indexing and pass explicit `class_ids` and `names` whenever the model uses a fixed class vocabulary.

## Example

See [`build_detections.py`](../../../examples/outputs/build_detections.py) for an executable example.

## Tests

See [`test_detect.py`](../../../test/outputs/test_detect.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.outputs.build_detections."""

from PIL import Image
from klygo import outputs

image = Image.new("RGB", (80, 60), "white")
frames = outputs.build_detections(
    [image],
    [
        {
            "boxes": [[10, 10, 40, 35], [45, 15, 75, 50]],
            "scores": [0.92, 0.81],
            "labels": ["item", "part"],
            "class_ids": [3, 7],
        }
    ],
    metadata={"backend": "custom", "threshold": 0.25},
    names={3: "item", 7: "part"},
)

frame = frames[0]
print(frame)
print("Box 0:", frame[0].id, frame[0].class_id, frame[0].label)
print("Frame names:", frame.names)

assert len(frames) == 1
assert frame[0].id == 0 and frame[0].class_id == 3
assert frame[1].id == 1 and frame[1].class_id == 7
assert frame.names == {3: "item", 7: "part"}
assert frame.metadata["backend"] == "custom"
```
