# `preprocessing.rotate`

Defined in `transforms.py`.

```python
preprocessing.rotate(angle: 'float', expand: 'bool' = False, fill: 'Any' = 0, interpolation: 'str' = 'bilinear') -> 'Operation'
```

Build a rotate image preprocessing operation.

**Parameters**

- `angle` (float): Rotation angle in degrees.
- `expand` (bool; default `False`): Value passed to the operation or evaluator.
- `fill` (Any; default `0`): Value passed to the operation or evaluator.
- `interpolation` (str; default `'bilinear'`): Resampling algorithm.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.rotate(5)
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/rotate.py) and see the [module index](../README.md).
