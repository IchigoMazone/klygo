# `preprocessing.random_apply`

Defined in `transforms.py`.

```python
preprocessing.random_apply(operation: 'Union[Operation, Compose]', probability: 'float' = 0.5) -> 'Operation'
```

Build a random apply image preprocessing operation.

**Parameters**

- `operation` (Union[Operation, Compose]): Value passed to the operation or evaluator.
- `probability` (float; default `0.5`): Probability that a random operation runs.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.random_apply(p.invert(), probability=1)
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/random_apply.py) and see the [module index](../README.md).
