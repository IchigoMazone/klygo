# `graphics.cell`

```python
graphics.cell(source, *, width=None, height=None, align=None, offset=None, title=None)
```

Create a per-cell layout specification for `compose` and `grid`.

```python
from PIL import Image
from klygo import graphics

c = graphics.cell(Image.new("RGB", (64, 48), "white"), width=120, align="top-left", offset=(4, 4), title="Sample")
print(c.width, c.align)
```

See [`examples/graphics/cell.py`](../../../examples/graphics/cell.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
