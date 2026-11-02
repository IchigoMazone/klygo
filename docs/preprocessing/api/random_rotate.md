# `preprocessing.random_rotate`

Defined in `transforms.py`.

```python
preprocessing.random_rotate(angle_range: 'Tuple[float, float]' = (-10, 10), expand: 'bool' = False) -> 'Operation'
```

Build a random rotate image preprocessing operation.

**Parameters**

- `angle_range` (Tuple[float, float]; default `(-10, 10)`): Value passed to the operation or evaluator.
- `expand` (bool; default `False`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.random_rotate()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/random_rotate.py) and see the [module index](../README.md).
