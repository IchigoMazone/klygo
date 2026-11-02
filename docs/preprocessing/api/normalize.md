# `preprocessing.normalize`

Defined in `transforms.py`.

```python
preprocessing.normalize(mean: 'Optional[Sequence[float]]' = None, std: 'Optional[Sequence[float]]' = None, scale: 'Optional[float]' = None) -> 'Operation'
```

Build a normalize image preprocessing operation.

**Parameters**

- `mean` (Optional[Sequence[float]]; default `None`): Value passed to the operation or evaluator.
- `std` (Optional[Sequence[float]]; default `None`): Value passed to the operation or evaluator.
- `scale` (Optional[float]; default `None`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.normalize()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/normalize.py) and see the [module index](../README.md).
