# `graphics.grid`

```python
graphics.grid(items, *, rows=None, cols=None, cell_size=None, gap=16, padding=24, bg_color='white', title=None, show=True, save_path=None, dpi=300, return_figure=False)
```

Arrange images or `compose` cells into a `rows x cols` grid figure.

```python
from PIL import Image
from klygo import graphics

items = graphics.compose([graphics.cell(Image.new("RGB", (64, 48), "red")), graphics.cell(Image.new("RGB", (64, 48), "blue"))])
graphics.grid(items, cols=2, show=False)
```

See [`examples/graphics/grid.py`](../../../examples/graphics/grid.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
