# `preprocessing.pad`

Defined in `transforms.py`.

```python
preprocessing.pad(padding: 'Union[int, Sequence[int]]', color: 'Any' = 0) -> 'Operation'
```

Build a pad image preprocessing operation.

**Parameters**

- `padding` (Union[int, Sequence[int]]): Pixels added around the image.
- `color` (Any; default `0`): Fill color.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.pad(2)
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/pad.py) and see the [module index](../README.md).
