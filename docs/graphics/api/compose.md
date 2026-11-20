# `graphics.compose`

```python
graphics.compose(cells)
```

Normalize heterogeneous `GridCell` items into resolved RGB images.

```python
from PIL import Image
from klygo import graphics

items = graphics.compose([graphics.cell(Image.new("RGB", (64, 48), "white"), width=100)])
print(len(items), items[0].source.size)
```

See [`examples/graphics/compose.py`](../../../examples/graphics/compose.py) and [`test/graphics/test_layout_and_charts.py`](../../../test/graphics/test_layout_and_charts.py).
