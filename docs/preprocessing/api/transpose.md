# `preprocessing.transpose`

Defined in `transforms.py`.

```python
preprocessing.transpose(method: 'str') -> 'Operation'
```

Build a transpose image preprocessing operation.

**Parameters**

- `method` (str): Named conversion or algorithm.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.transpose("rotate_90")
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/transpose.py) and see the [module index](../README.md).
