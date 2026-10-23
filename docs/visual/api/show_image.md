# `visual.show_image`

```python
visual.show_image(
    image,
    title="Image",
    width=None,
    backend="auto",
    figsize=(10, 10),
)
```

Display a PIL image or RGB NumPy array using a notebook, desktop, Matplotlib, or OpenCV backend.

## Contract

Accepts a `PIL.Image.Image` or an RGB `numpy.ndarray` (2D grayscale or 3D RGB/RGBA) and displays it without mutating the caller's image. When `width` is a positive integer, a proportionally scaled copy is created using bilinear resampling (`height = max(1, int(h_orig * (width / w_orig)))`) prior to rendering.

Backend selection follows `backend`:
- `"auto"` (default): Checks whether execution is inside an interactive Jupyter, Google Colab, or Kaggle kernel (`ZMQInteractiveShell`, `Shell`, or `google.colab`). If so, displays the image inline via `IPython.display.display(pil_img)`. Otherwise, opens the system viewer via `pil_img.show(title=title)`.
- `"matplotlib"` (case-insensitive): Creates a Matplotlib figure of size `figsize`, renders the image with axes hidden and `title` set, and calls `plt.show()`.
- Any other string (such as `"opencv"` or `"cv2"`): Converts the RGB image to BGR, opens a window via `cv2.imshow(title, img_cv)`, waits indefinitely for a key press (`cv2.waitKey(0)`), and closes all OpenCV windows (`cv2.destroyAllWindows()`).

## Parameters

- `image`: `PIL.Image.Image | numpy.ndarray` — image to display; NumPy arrays use RGB channel order.
- `title`: `str`, default `"Image"` — window title or Matplotlib plot title.
- `width`: `int | None`, default `None` — optional positive target width in pixels for proportional resizing before display; non-positive values or `None` leave dimensions unchanged.
- `backend`: `str`, default `"auto"` — rendering backend (`"auto"`, `"matplotlib"`, or `"opencv"`).
- `figsize`: `tuple[int, int]`, default `(10, 10)` — Matplotlib figure dimensions `(width_inches, height_inches)` used when `backend="matplotlib"`.

## Returns

`None` — displays the image as a side effect without returning a value.

## Errors and edge cases

- Raises `ImportError` / `ModuleNotFoundError` if `backend="matplotlib"` or OpenCV is selected and the corresponding library is not installed.
- Raises `TypeError` or `AttributeError` if `image` is neither a `PIL.Image.Image` nor a valid `numpy.ndarray`.
- When using the OpenCV backend in headless environments without a display server, `cv2.imshow` may fail or block waiting for input; prefer `backend="matplotlib"` with a non-interactive backend (`Agg`) in automated scripts and tests.

## AI usage guidance

Prefer `from klygo import visual` followed by `visual.show_image(...)`. Use `backend="auto"` in tutorials and notebooks, and `backend="matplotlib"` when a figure title or consistent non-blocking test rendering is required.

## Example

See [`show_image.py`](../../../examples/visual/show_image.py) for an executable example.

## Tests

See [`test_visual.py`](../../../test/visual/test_visual.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.visual.show_image."""

import matplotlib
import numpy as np
from klygo import visual

matplotlib.use("Agg", force=True)
image = np.full((40, 60, 3), (30, 100, 180), dtype=np.uint8)
result = visual.show_image(
    image,
    title="Sample",
    width=120,
    backend="matplotlib",
    figsize=(6, 4),
)

print("Displayed a 120-pixel-wide image")
assert result is None
assert image.shape == (40, 60, 3)
```
