# `preprocessing.colorize`

Defined in `transforms.py`.

```python
preprocessing.colorize(black: 'Any', white: 'Any', mid: 'Any' = None) -> 'Operation'
```

Build a colorize image preprocessing operation.

**Parameters**

- `black` (Any): Value passed to the operation or evaluator.
- `white` (Any): Value passed to the operation or evaluator.
- `mid` (Any; default `None`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.colorize("black", "white")
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/colorize.py) and see the [module index](../README.md).
