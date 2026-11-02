# `preprocessing.resize`

Defined in `transforms.py`.

```python
preprocessing.resize(size: 'Size', keep_ratio: 'bool' = False, interpolation: 'str' = 'bilinear') -> 'Operation'
```

Build a resize image preprocessing operation.

**Parameters**

- `size` (Size): Target width and height.
- `keep_ratio` (bool; default `False`): Preserve input aspect ratio when true.
- `interpolation` (str; default `'bilinear'`): Resampling algorithm.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.resize((32, 24))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/resize.py) and see the [module index](../README.md).
