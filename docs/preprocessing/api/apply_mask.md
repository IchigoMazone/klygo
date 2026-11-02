# `preprocessing.apply_mask`

Defined in `transforms.py`.

```python
preprocessing.apply_mask(mask: 'Any') -> 'Operation'
```

Build a apply mask image preprocessing operation.

**Parameters**

- `mask` (Any): Mask image.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.apply_mask(Image.new("L", image.size, 128))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/apply_mask.py) and see the [module index](../README.md).
