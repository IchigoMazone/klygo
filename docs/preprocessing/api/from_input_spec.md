# `preprocessing.from_input_spec`

Defined in `transforms.py`.

```python
preprocessing.from_input_spec(input_spec: 'Any') -> 'Compose'
```

Build a preprocessing pipeline from a model input specification.

**Parameters**

- `input_spec` (Any): Value passed to the operation or evaluator.

**Returns and effects:** Operation; `from_input_spec` returns Compose. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Invalid options or shapes can raise `TypeError` or `ValueError` while building or executing the operation. Pillow, OpenCV, or Torch may report unsupported pixel data.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
pipeline = p.from_input_spec({})
print(type(pipeline).__name__, len(pipeline))
```

Run [the standalone example](../../../examples/preprocessing/from_input_spec.py) and see the [module index](../README.md).
