# `visual.draw_bboxes`

```python
visual.draw_bboxes(
    image,
    bboxes,
    labels=None,
    scores=None,
    thickness=None,
    line_width=None,
    font_size=None,
    box_color=None,
    text_color=None,
    show_labels=True,
    show_scores=True,
    **kwargs,
)
```

Draw `[xmin, ymin, xmax, ymax]` bounding boxes and optional caption badges on a copy of an image.

## Contract

Creates an annotated copy of `image` without mutating the caller's input. Accepts a `PIL.Image.Image`, an RGB `numpy.ndarray` (2D grayscale or 3D RGB/RGBA), or a `media.LazyImage` (materialized via `.to_pil(cache=False)`). When `image` is a `numpy.ndarray`, the returned annotated image is also converted back to a `numpy.ndarray`; otherwise, a `PIL.Image.Image` is returned.

Rendering proceeds in two passes:
1. **Pass 1 (Box outlines)**: Every box in `bboxes` is drawn using `line_width` (which takes precedence over `thickness` and the legacy `width` keyword in `kwargs`, or scales automatically from image dimensions when omitted). Each unique label receives a deterministic RGB color from the 20-color Ultralytics palette unless `box_color` overrides all boxes.
2. **Pass 2 (Caption badges)**: When `show_labels` or `show_scores` produces a non-empty caption, a filled badge is placed above the top-left corner of the box—or inside the box when near the top image edge or when colliding with a previously placed badge.

## Parameters

- `image`: `PIL.Image.Image | numpy.ndarray` — source image to annotate (NumPy arrays use RGB channel order).
- `bboxes`: ` list[list[float] | tuple[float, float, float, float]]` — sequence of bounding boxes in `[xmin, ymin, xmax, ymax]` pixel coordinates.
- `labels`: `list[str] | None`, default `None` — class label strings matched to `bboxes` by index.
- `scores`: `list[float] | None`, default `None` — confidence scores matched to `bboxes` by index and formatted to two decimal places (`0.92`).
- `thickness`: `int | None`, default `None` — border thickness in pixels (fallback when `line_width` is not set).
- `line_width`: `int | None`, default `None` — border width in pixels; takes precedence over `thickness` and `kwargs.get("width")`.
- `font_size`: `int | None`, default `None` — caption font size in pixels; auto-scaled from image dimensions when `None`.
- `box_color`: `tuple[int, int, int] | None`, default `None` — fixed RGB `(R, G, B)` tuple overriding palette colors for all boxes and badge backgrounds.
- `text_color`: `tuple[int, int, int] | None`, default `None` — RGB `(R, G, B)` tuple for caption text (defaults to `(255, 255, 255)`).
- `show_labels`: `bool`, default `True` — include class labels in caption badges.
- `show_scores`: `bool`, default `True` — include formatted confidence scores in caption badges.
- `**kwargs`: `Any` — supports legacy `width` keyword as a fallback border width.

## Returns

`PIL.Image.Image | numpy.ndarray` — annotated image copy matching the input representation (`numpy.ndarray` if `image` was a NumPy array, otherwise `PIL.Image.Image`).

## Errors and edge cases

- Raises `ValueError` or `TypeError` when any element of `bboxes` cannot be unpacked into four numeric coordinates `xmin, ymin, xmax, ymax`.
- Raises `AttributeError` or `TypeError` if `image` is neither a `PIL.Image.Image`, a `numpy.ndarray`, nor an object providing `.to_pil()` / `.copy()`.
- Missing system TrueType fonts (`arial.ttf`, `DejaVuSans.ttf`, `calibri.ttf`, `SegoeUI.ttf`) fall back gracefully to `PIL.ImageFont.load_default()`.

## AI usage guidance

Prefer `from klygo import visual` followed by `visual.draw_bboxes(...)`. Pass RGB arrays rather than OpenCV BGR arrays so colors render accurately.

## Example

See [`draw_bboxes.py`](../../../examples/visual/draw_bboxes.py) for an executable example.

## Tests

See [`test_visual.py`](../../../test/visual/test_visual.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.visual.draw_bboxes."""

import numpy as np
from klygo import visual

image = np.full((80, 120, 3), 255, dtype=np.uint8)
annotated = visual.draw_bboxes(
    image,
    [[10, 12, 80, 65]],
    labels=["sample"],
    scores=[0.92],
    box_color=(220, 30, 30),
    line_width=2,
)

print("Input unchanged:", bool(np.all(image == 255)))
print("Output shape:", annotated.shape)

assert isinstance(annotated, np.ndarray)
assert annotated.shape == (80, 120, 3)
assert np.all(image == 255)
assert not np.all(annotated == 255)
```
