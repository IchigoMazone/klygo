# `preprocessing.apply`

Defined in `core.py`.

```python
preprocessing.apply(image: 'Any', operation_or_pipeline: 'Union[Operation, Compose]') -> 'Any'
```

Create or execute apply for an image preprocessing pipeline.

**Parameters**

- `image` (Any): Input PIL, NumPy, or lazy image.
- `operation_or_pipeline` (Union[Operation, Compose]): Value passed to the operation or evaluator.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.apply(image, p.resize((32, 24))).size)
```

Run [the standalone example](../../../examples/preprocessing/apply.py) and see the [module index](../README.md).
