# `preprocessing.Predicate`

Defined in `core.py`.

```python
preprocessing.Predicate(evaluator: 'Callable[[Any], bool]', description: 'str' = 'predicate') -> 'None'
```

Create or execute Predicate for an image preprocessing pipeline.

**Parameters**

- `evaluator` (Callable[[Any], bool]): Value passed to the operation or evaluator.
- `description` (str; default `'predicate'`): Value passed to the operation or evaluator.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
condition = p.Predicate(lambda value: value.size[0] > 32)
print(condition(image))
```

Run [the standalone example](../../../examples/preprocessing/Predicate.py) and see the [module index](../README.md).
