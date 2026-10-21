# `preprocessing.remove_alpha`

Defined in `transforms.py`.

```python
preprocessing.remove_alpha(background: 'Any' = (255, 255, 255)) -> 'Operation'
```

Build a remove alpha image preprocessing operation.

**Parameters**

- `background` (Any; default `(255, 255, 255)`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.remove_alpha()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/remove_alpha.py) and see the [module index](../README.md).
