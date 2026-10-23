# `preprocessing.convert_color`

Defined in `transforms.py`.

```python
preprocessing.convert_color(source: 'str', target: 'Optional[str]' = None) -> 'Operation'
```

Build a convert color image preprocessing operation.

**Parameters**

- `source` (str): Input color representation.
- `target` (Optional[str]; default `None`): Output color representation.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.convert_color("HSV")
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/convert_color.py) and see the [module index](../README.md).
