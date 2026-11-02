# `preprocessing.sobel`

Defined in `transforms.py`.

```python
preprocessing.sobel(dx: 'int' = 1, dy: 'int' = 0, kernel: 'int' = 3) -> 'Operation'
```

Build a sobel image preprocessing operation.

**Parameters**

- `dx` (int; default `1`): Value passed to the operation or evaluator.
- `dy` (int; default `0`): Value passed to the operation or evaluator.
- `kernel` (int; default `3`): Filter or morphology kernel.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.sobel()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/sobel.py) and see the [module index](../README.md).
