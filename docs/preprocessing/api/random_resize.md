# `preprocessing.random_resize`

Defined in `transforms.py`.

```python
preprocessing.random_resize(min_size: 'Size', max_size: 'Size') -> 'Operation'
```

Build a random resize image preprocessing operation.

**Parameters**

- `min_size` (Size): Value passed to the operation or evaluator.
- `max_size` (Size): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.random_resize((32, 24), (40, 30))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/random_resize.py) and see the [module index](../README.md).
