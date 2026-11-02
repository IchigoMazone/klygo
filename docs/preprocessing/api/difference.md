# `preprocessing.difference`

Defined in `transforms.py`.

```python
preprocessing.difference(other: 'Any') -> 'Operation'
```

Build a difference image preprocessing operation.

**Parameters**

- `other` (Any): Second image or value.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.difference(Image.new("RGB", image.size, "blue"))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/difference.py) and see the [module index](../README.md).
