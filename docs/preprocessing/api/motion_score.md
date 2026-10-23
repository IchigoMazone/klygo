# `preprocessing.motion_score`

Defined in `metrics.py`.

```python
preprocessing.motion_score(previous: 'Any', preview_size: 'Tuple[int, int]' = (160, 160)) -> 'Metric'
```

Measure or prepare the motion score image statistic.

**Parameters**

- `previous` (Any): Value passed to the operation or evaluator.
- `preview_size` (Tuple[int, int]; default `(160, 160)`): Resize used for inexpensive metric evaluation.

**Returns and effects:** Metric or evaluated statistic. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Unsupported image data or algorithm names can raise `TypeError` or `ValueError` during evaluation.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.motion_score(image)(image))
```

Run [the standalone example](../../../examples/preprocessing/motion_score.py) and see the [module index](../README.md).
