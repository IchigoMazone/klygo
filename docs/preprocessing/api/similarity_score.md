# `preprocessing.similarity_score`

Defined in `metrics.py`.

```python
preprocessing.similarity_score(other: 'Any', method: 'str' = 'phash', preview_size: 'Tuple[int, int]' = (32, 32)) -> 'Metric'
```

Measure or prepare the similarity score image statistic.

**Parameters**

- `other` (Any): Second image or value.
- `method` (str; default `'phash'`): Named conversion or algorithm.
- `preview_size` (Tuple[int, int]; default `(32, 32)`): Resize used for inexpensive metric evaluation.

**Returns and effects:** Metric or evaluated statistic. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Unsupported image data or algorithm names can raise `TypeError` or `ValueError` during evaluation.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.similarity_score(image)(image))
```

Run [the standalone example](../../../examples/preprocessing/similarity_score.py) and see the [module index](../README.md).
