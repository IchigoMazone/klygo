# `preprocessing.inspect`

Defined in `core.py`.

```python
preprocessing.inspect(pipeline: 'Union[Operation, Compose, Any]') -> 'List[Dict[str, Any]]'
```

Create or execute inspect for an image preprocessing pipeline.

**Parameters**

- `pipeline` (Union[Operation, Compose, Any]): Operation or composed pipeline to inspect or optimize.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
print(p.inspect(p.resize((32, 24))))
```

Run [the standalone example](../../../examples/preprocessing/inspect.py) and see the [module index](../README.md).
