# `preprocessing.crop`

Defined in `transforms.py`.

```python
preprocessing.crop(box: 'Sequence[int]') -> 'Operation'
```

Build a crop image preprocessing operation.

**Parameters**

- `box` (Sequence[int]): XYXY pixel crop coordinates.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.crop((5, 5, 50, 40))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/crop.py) and see the [module index](../README.md).
