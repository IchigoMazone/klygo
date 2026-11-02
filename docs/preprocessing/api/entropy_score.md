# `preprocessing.entropy_score`

Defined in `metrics.py`.

```python
preprocessing.entropy_score(preview_size: 'Tuple[int, int]' = (160, 160)) -> 'Metric'
```

Measure or prepare the entropy score image statistic.

**Parameters**

- `preview_size` (Tuple[int, int]; default `(160, 160)`): Resize used for inexpensive metric evaluation.

**Returns and effects:** Metric or evaluated statistic. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Unsupported image data or algorithm names can raise `TypeError` or `ValueError` during evaluation.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.entropy_score()(image))
```

Run [the standalone example](../../../examples/preprocessing/entropy_score.py) and see the [module index](../README.md).
