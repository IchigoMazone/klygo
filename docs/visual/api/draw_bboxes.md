# `visual.draw_bboxes`

```python
visual.draw_bboxes(image, bboxes, labels=None, scores=None, thickness=None,
                   line_width=None, font_size=None, box_color=None,
                   text_color=None, show_labels=True, show_scores=True, **kwargs)
```

Draw pixel-coordinate `[xmin, ymin, xmax, ymax]` boxes on a copy of a PIL image
or RGB NumPy array. `labels` and `scores` are matched to boxes by position.
`line_width` takes precedence over `thickness`; `box_color` and `text_color`
override RGB colors. The two `show_*` flags control caption content. Optional
`font_size` controls caption size. The legacy `width` keyword in `kwargs` is a
border-width fallback. Returns the same image representation as input. A malformed
box or an unsupported image type raises an error from PIL or the coordinate cast.

```python
import numpy as np
from klygo import visual

image = np.full((40, 60, 3), 255, dtype=np.uint8)
annotated = visual.draw_bboxes(image, [[5, 5, 30, 25]], labels=["item"])
print(annotated.shape)
```

See [the runnable example](../../../examples/visual/draw_bboxes.py) and
[`show_image`](show_image.md) for displaying the result.
