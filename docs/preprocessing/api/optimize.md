# `preprocessing.optimize`

Defined in `core.py`.

```python
preprocessing.optimize(pipeline: 'Union[Operation, Compose]') -> 'Compose'
```

Create or execute optimize for an image preprocessing pipeline.

**Parameters**

- `pipeline` (Union[Operation, Compose]): Operation or composed pipeline to inspect or optimize.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
pipeline = p.optimize(p.compose(p.resize((32, 24)), p.resize((16, 16))))
print(len(pipeline))
```

Run [the standalone example](../../../examples/preprocessing/optimize.py) and see the [module index](../README.md).
