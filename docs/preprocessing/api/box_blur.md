# `preprocessing.box_blur`

Defined in `transforms.py`.

```python
preprocessing.box_blur(radius: 'float' = 1) -> 'Operation'
```

Build a box blur image preprocessing operation.

**Parameters**

- `radius` (float; default `1`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.box_blur()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/box_blur.py) and see the [module index](../README.md).
