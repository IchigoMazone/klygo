# `preprocessing.composite`

Defined in `transforms.py`.

```python
preprocessing.composite(foreground: 'Any', background: 'Any', mask: 'Any') -> 'Operation'
```

Build a composite image preprocessing operation.

**Parameters**

- `foreground` (Any): Value passed to the operation or evaluator.
- `background` (Any): Value passed to the operation or evaluator.
- `mask` (Any): Mask image.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.composite(image, Image.new("RGB", image.size, "blue"), Image.new("L", image.size, 128))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/composite.py) and see the [module index](../README.md).
