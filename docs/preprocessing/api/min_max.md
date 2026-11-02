# `preprocessing.min_max`

Defined in `metrics.py`.

```python
preprocessing.min_max(image: 'Any' = None, channel: 'Optional[int]' = None) -> 'Any'
```

Measure or prepare the min max image statistic.

**Parameters**

- `image` (Any; default `None`): Input PIL, NumPy, or lazy image.
- `channel` (Optional[int]; default `None`): Value passed to the operation or evaluator.

**Returns and effects:** Metric or evaluated statistic. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Unsupported image data or algorithm names can raise `TypeError` or `ValueError` during evaluation.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.min_max(image))
```

Run [the standalone example](../../../examples/preprocessing/min_max.py) and see the [module index](../README.md).
