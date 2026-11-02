# `preprocessing.remove_batch_dimension`

Defined in `transforms.py`.

```python
preprocessing.remove_batch_dimension() -> 'Operation'
```

Build a remove batch dimension image preprocessing operation.

**Parameters**

- No parameters.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.remove_batch_dimension()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/remove_batch_dimension.py) and see the [module index](../README.md).
