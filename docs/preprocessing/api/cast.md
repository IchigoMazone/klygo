# `preprocessing.cast`

Defined in `transforms.py`.

```python
preprocessing.cast(dtype: 'str' = 'float32') -> 'Operation'
```

Build a cast image preprocessing operation.

**Parameters**

- `dtype` (str; default `'float32'`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.cast()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/cast.py) and see the [module index](../README.md).
