# `preprocessing.bilateral_filter`

Defined in `transforms.py`.

```python
preprocessing.bilateral_filter(diameter: 'int' = 5, sigma_color: 'float' = 50, sigma_space: 'float' = 50) -> 'Operation'
```

Build a bilateral filter image preprocessing operation.

**Parameters**

- `diameter` (int; default `5`): Value passed to the operation or evaluator.
- `sigma_color` (float; default `50`): Value passed to the operation or evaluator.
- `sigma_space` (float; default `50`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.bilateral_filter()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/bilateral_filter.py) and see the [module index](../README.md).
