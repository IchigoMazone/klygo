# `preprocessing.convolve`

Defined in `transforms.py`.

```python
preprocessing.convolve(kernel: 'Sequence[Sequence[float]]', scale: 'Optional[float]' = None, offset: 'float' = 0) -> 'Operation'
```

Build a convolve image preprocessing operation.

**Parameters**

- `kernel` (Sequence[Sequence[float]]): Filter or morphology kernel.
- `scale` (Optional[float]; default `None`): Value passed to the operation or evaluator.
- `offset` (float; default `0`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.convolve([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/convolve.py) and see the [module index](../README.md).
