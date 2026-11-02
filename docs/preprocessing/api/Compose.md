# `preprocessing.compose`

Defined in `core.py`.

```python
preprocessing.compose(*operations: 'Any') -> 'Compose'
```

Create or execute compose for an image preprocessing pipeline.

**Parameters**

- `operations` (Any): Ordered preprocessing operations to compose or execute.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
pipeline = p.compose(p.resize((32, 24)), p.grayscale())
print(pipeline(image).size)
```

Run [the standalone example](../../../examples/preprocessing/compose.py) and see the [module index](../README.md).
