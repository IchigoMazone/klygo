# `preprocessing.clahe`

Defined in `transforms.py`.

```python
preprocessing.clahe(clip_limit: 'float' = 2.0, grid_size: 'Tuple[int, int]' = (8, 8)) -> 'Operation'
```

Build a clahe image preprocessing operation.

**Parameters**

- `clip_limit` (float; default `2.0`): Value passed to the operation or evaluator.
- `grid_size` (Tuple[int, int]; default `(8, 8)`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.clahe()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/clahe.py) and see the [module index](../README.md).
