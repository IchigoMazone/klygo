# `visual.show_image`

```python
visual.show_image(image, title="Image", width=None, backend="auto", figsize=(10, 10))
```

Display a PIL image or RGB NumPy array and return `None`. A positive `width`
rescales the image proportionally. `auto` displays inline in notebooks and opens
the desktop viewer elsewhere; `matplotlib` uses pyplot with `figsize` in inches;
other backend strings select OpenCV and block until a key is pressed. Missing
optional Matplotlib or OpenCV packages raise `ImportError` when selected.

```python
import matplotlib
from PIL import Image
from klygo import visual

matplotlib.use("Agg", force=True)
visual.show_image(Image.new("RGB", (30, 20)), backend="matplotlib")
```

See [the runnable example](../../../examples/visual/show_image.py) and
[`draw_bboxes`](draw_bboxes.md) for annotation.
