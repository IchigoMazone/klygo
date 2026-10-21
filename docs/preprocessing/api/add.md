# `preprocessing.add`

Defined in `transforms.py`.

```python
preprocessing.add(other: 'Any', scale: 'float' = 1.0) -> 'Operation'
```

Build a add image preprocessing operation.

**Parameters**

- `other` (Any): Second image or value.
- `scale` (float; default `1.0`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.add(Image.new("RGB", image.size, "blue"))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/add.py) and see the [module index](../README.md).
