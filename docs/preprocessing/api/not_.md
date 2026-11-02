# `preprocessing.not_`

Defined in `core.py`.

```python
preprocessing.not_(condition: 'Predicate') -> 'Predicate'
```

Create or execute not  for an image preprocessing pipeline.

**Parameters**

- `condition` (Predicate): Predicate to invert.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
condition = p.not_(p.brightness_score() < 0)
print(condition(image))
```

Run [the standalone example](../../../examples/preprocessing/not_.py) and see the [module index](../README.md).
