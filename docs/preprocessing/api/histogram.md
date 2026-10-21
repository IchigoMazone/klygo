# `preprocessing.histogram`

Defined in `metrics.py`.

```python
preprocessing.histogram(image: 'Any' = None, channel: 'Optional[int]' = None, bins: 'int' = 256) -> 'Any'
```

Measure or prepare the histogram image statistic.

**Parameters**

- `image` (Any; default `None`): Input PIL, NumPy, or lazy image.
- `channel` (Optional[int]; default `None`): Value passed to the operation or evaluator.
- `bins` (int; default `256`): Value passed to the operation or evaluator.

**Returns and effects:** Metric or evaluated statistic. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Unsupported image data or algorithm names can raise `TypeError` or `ValueError` during evaluation.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(len(p.histogram(image)))
```

Run [the standalone example](../../../examples/preprocessing/histogram.py) and see the [module index](../README.md).
