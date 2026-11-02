# `preprocessing.random_erasing`

Defined in `transforms.py`.

```python
preprocessing.random_erasing(probability: 'float' = 0.5, scale: 'Tuple[float, float]' = (0.02, 0.33), value: 'Any' = 0) -> 'Operation'
```

Build a random erasing image preprocessing operation.

**Parameters**

- `probability` (float; default `0.5`): Probability that a random operation runs.
- `scale` (Tuple[float, float]; default `(0.02, 0.33)`): Value passed to the operation or evaluator.
- `value` (Any; default `0`): Input value or numeric setting.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.random_erasing()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/random_erasing.py) and see the [module index](../README.md).
