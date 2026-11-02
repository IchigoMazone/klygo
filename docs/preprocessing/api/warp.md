# `preprocessing.warp`

Defined in `transforms.py`.

```python
preprocessing.warp(matrix: 'Sequence[float]', size: 'Optional[Size]' = None, fill: 'Any' = 0) -> 'Operation'
```

Build a warp image preprocessing operation.

**Parameters**

- `matrix` (Sequence[float]): Affine or perspective transform coefficients.
- `size` (Optional[Size]; default `None`): Target width and height.
- `fill` (Any; default `0`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.warp((1, 0, 0, 0, 1, 0, 0, 0))
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/warp.py) and see the [module index](../README.md).
