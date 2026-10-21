# `preprocessing.select_channels`

Defined in `transforms.py`.

```python
preprocessing.select_channels(channels: 'Sequence[int]') -> 'Operation'
```

Build a select channels image preprocessing operation.

**Parameters**

- `channels` (Sequence[int]): Channel count or selected channel indexes.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.select_channels((0, 1))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/select_channels.py) and see the [module index](../README.md).
