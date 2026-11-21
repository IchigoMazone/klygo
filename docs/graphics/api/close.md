# `graphics.close`

```python
graphics.close(*figures)
```

Close one or more Matplotlib figures to free resources without directly importing `matplotlib.pyplot`.

```python
from klygo import graphics

fig = graphics.bars([{"strategy": "random", "mAP50-95": 0.5}], show=False, return_figure=True)
graphics.close(fig)
```

See [`examples/graphics/close.py`](../../../examples/graphics/close.py) and [`test/graphics/test_graphics.py`](../../../test/graphics/test_graphics.py).
