# `preprocessing.perceptual_hash`

Defined in `metrics.py`.

```python
preprocessing.perceptual_hash(image: 'Any' = None, method: 'str' = 'phash') -> 'Any'
```

Measure or prepare the perceptual hash image statistic.

**Parameters**

- `image` (Any; default `None`): Input PIL, NumPy, or lazy image.
- `method` (str; default `'phash'`): Named conversion or algorithm.

**Returns and effects:** Metric or evaluated statistic. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Unsupported image data or algorithm names can raise `TypeError` or `ValueError` during evaluation.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.perceptual_hash(image))
```

Run [the standalone example](../../../examples/preprocessing/perceptual_hash.py) and see the [module index](../README.md).
