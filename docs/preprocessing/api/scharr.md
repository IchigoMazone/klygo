# `preprocessing.scharr`

Defined in `transforms.py`.

```python
preprocessing.scharr(dx: 'int' = 1, dy: 'int' = 0) -> 'Operation'
```

Build a scharr image preprocessing operation.

**Parameters**

- `dx` (int; default `1`): Value passed to the operation or evaluator.
- `dy` (int; default `0`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.scharr()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/scharr.py) and see the [module index](../README.md).
