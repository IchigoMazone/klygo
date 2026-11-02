# `preprocessing.canny`

Defined in `transforms.py`.

```python
preprocessing.canny(low: 'float' = 100, high: 'float' = 200, aperture_size: 'int' = 3) -> 'Operation'
```

Build a canny image preprocessing operation.

**Parameters**

- `low` (float; default `100`): Value passed to the operation or evaluator.
- `high` (float; default `200`): Value passed to the operation or evaluator.
- `aperture_size` (int; default `3`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.canny()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/canny.py) and see the [module index](../README.md).
