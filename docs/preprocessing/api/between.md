# `preprocessing.between`

Defined in `core.py`.

```python
preprocessing.between(metric: 'Metric', minimum: 'Any', maximum: 'Any') -> 'Predicate'
```

Create or execute between for an image preprocessing pipeline.

**Parameters**

- `metric` (Metric): Metric used to build a condition.
- `minimum` (Any): Lower bound or threshold.
- `maximum` (Any): Upper bound.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
condition = p.between(p.brightness_score(), 0, 255)
print(condition(image))
```

Run [the standalone example](../../../examples/preprocessing/between.py) and see the [module index](../README.md).
