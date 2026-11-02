# `preprocessing.random_flip`

Defined in `transforms.py`.

```python
preprocessing.random_flip(probability: 'float' = 0.5, horizontal: 'bool' = True, vertical: 'bool' = False) -> 'Operation'
```

Build a random flip image preprocessing operation.

**Parameters**

- `probability` (float; default `0.5`): Probability that a random operation runs.
- `horizontal` (bool; default `True`): Apply horizontal orientation change.
- `vertical` (bool; default `False`): Apply vertical orientation change.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.random_flip()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/random_flip.py) and see the [module index](../README.md).
