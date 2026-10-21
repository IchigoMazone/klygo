# `preprocessing.color_jitter`

Defined in `transforms.py`.

```python
preprocessing.color_jitter(brightness: 'float' = 0, contrast: 'float' = 0, saturation: 'float' = 0, hue: 'float' = 0) -> 'Operation'
```

Build a color jitter image preprocessing operation.

**Parameters**

- `brightness` (float; default `0`): Value passed to the operation or evaluator.
- `contrast` (float; default `0`): Value passed to the operation or evaluator.
- `saturation` (float; default `0`): Value passed to the operation or evaluator.
- `hue` (float; default `0`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.color_jitter()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/color_jitter.py) and see the [module index](../README.md).
