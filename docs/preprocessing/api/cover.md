# `preprocessing.cover`

Defined in `transforms.py`.

```python
preprocessing.cover(size: 'Size', interpolation: 'str' = 'bilinear') -> 'Operation'
```

Build a cover image preprocessing operation.

**Parameters**

- `size` (Size): Target width and height.
- `interpolation` (str; default `'bilinear'`): Resampling algorithm.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.cover((32, 24))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/cover.py) and see the [module index](../README.md).
