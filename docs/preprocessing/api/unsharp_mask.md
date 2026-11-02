# `preprocessing.unsharp_mask`

Defined in `transforms.py`.

```python
preprocessing.unsharp_mask(radius: 'float' = 2, amount: 'float' = 1.0, threshold: 'int' = 0) -> 'Operation'
```

Build a unsharp mask image preprocessing operation.

**Parameters**

- `radius` (float; default `2`): Value passed to the operation or evaluator.
- `amount` (float; default `1.0`): Value passed to the operation or evaluator.
- `threshold` (int; default `0`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.unsharp_mask()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/unsharp_mask.py) and see the [module index](../README.md).
