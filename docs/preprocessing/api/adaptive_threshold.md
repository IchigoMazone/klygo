# `preprocessing.adaptive_threshold`

Defined in `transforms.py`.

```python
preprocessing.adaptive_threshold(method: 'str' = 'gaussian', block_size: 'int' = 11, constant: 'float' = 2, mode: 'str' = 'binary') -> 'Operation'
```

Build a adaptive threshold image preprocessing operation.

**Parameters**

- `method` (str; default `'gaussian'`): Named conversion or algorithm.
- `block_size` (int; default `11`): Value passed to the operation or evaluator.
- `constant` (float; default `2`): Value passed to the operation or evaluator.
- `mode` (str; default `'binary'`): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.adaptive_threshold()
result = operation(image)
print(operation.name, type(result).__name__)
```

Run [the standalone example](../../../examples/preprocessing/adaptive_threshold.py) and see the [module index](../README.md).
