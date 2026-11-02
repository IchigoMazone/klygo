# `preprocessing.all_of`

Defined in `core.py`.

```python
preprocessing.all_of(*conditions: 'Predicate') -> 'Predicate'
```

Create or execute all of for an image preprocessing pipeline.

**Parameters**

- `conditions` (Predicate): Predicates to combine.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
condition = p.all_of(p.brightness_score() > 0, p.contrast_score() >= 0)
print(condition(image))
```

Run [the standalone example](../../../examples/preprocessing/all_of.py) and see the [module index](../README.md).
