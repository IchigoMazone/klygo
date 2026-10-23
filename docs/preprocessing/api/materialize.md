# `preprocessing.materialize`

Defined in `core.py`.

```python
preprocessing.materialize(image: 'Any', output: 'str' = 'pil', cache: 'bool' = False) -> 'Any'
```

Create or execute materialize for an image preprocessing pipeline.

**Parameters**

- `image` (Any): Input PIL, NumPy, or lazy image.
- `output` (str; default `'pil'`): Materialization format such as PIL or array.
- `cache` (bool; default `False`): Whether to keep decoded pixels for later access.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.materialize(image, output="array").shape)
```

Run [the standalone example](../../../examples/preprocessing/materialize.py) and see the [module index](../README.md).
