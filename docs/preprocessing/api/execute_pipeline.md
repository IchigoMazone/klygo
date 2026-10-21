# `preprocessing.execute_pipeline`

Defined in `core.py`.

```python
preprocessing.execute_pipeline(value: 'Any', operations: 'Sequence[Operation]', trace: 'Optional[List[Dict[str, Any]]]' = None) -> 'Tuple[Any, List[Dict[str, Any]]]'
```

Create or execute execute pipeline for an image preprocessing pipeline.

**Parameters**

- `value` (Any): Input value or numeric setting.
- `operations` (Sequence[Operation]): Ordered preprocessing operations to compose or execute.
- `trace` (Optional[List[Dict[str, Any]]]; default `None`): Recorded geometry transformations.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
result, history = p.execute_pipeline(image, [p.resize((32, 24))])
print(result.size, len(history))
```

Run [the standalone example](../../../examples/preprocessing/execute_pipeline.py) and see the [module index](../README.md).
