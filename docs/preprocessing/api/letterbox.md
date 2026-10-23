# `preprocessing.letterbox`

Defined in `transforms.py`.

```python
preprocessing.letterbox(size: 'Size', color: 'Any' = (114, 114, 114), stride: 'Optional[int]' = None) -> 'Operation'
```

Build a letterbox image preprocessing operation.

**Parameters**

- `size` (Size): Target width and height.
- `color` (Any; default `(114, 114, 114)`): Fill color.
- `stride` (Optional[int]; default `None`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.letterbox((32, 24))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/letterbox.py) and see the [module index](../README.md).
