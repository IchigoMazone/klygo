# `preprocessing.register`

Defined in `core.py`.

```python
preprocessing.register(name: 'str', executor: 'Callable[..., Any]') -> 'None'
```

Create or execute register for an image preprocessing pipeline.

**Parameters**

- `name` (str): Public name used in operation descriptions and traces.
- `executor` (Callable[..., Any]): Callable implementation registered under a name.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
p.register("identity_example", lambda value, operation, trace: value)
print(p.Operation("identity_example")(image).size)
```

Run [the standalone example](../../../examples/preprocessing/register.py) and see the [module index](../README.md).
