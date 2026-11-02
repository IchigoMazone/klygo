# `preprocessing.Metric`

Defined in `core.py`.

```python
preprocessing.Metric(name: 'str', evaluator: 'Callable[[Any], Any]', params: 'Optional[Dict[str, Any]]' = None) -> 'None'
```

Create or execute Metric for an image preprocessing pipeline.

**Parameters**

- `name` (str): Public name used in operation descriptions and traces.
- `evaluator` (Callable[[Any], Any]): Value passed to the operation or evaluator.
- `params` (Optional[Dict[str, Any]]; default `None`): Operation parameters passed to the executor.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
metric = p.Metric("width", lambda value: value.size[0])
print(metric(image))
```

Run [the standalone example](../../../examples/preprocessing/Metric.py) and see the [module index](../README.md).
