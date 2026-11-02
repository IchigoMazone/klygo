# `preprocessing.Operation`

Defined in `core.py`.

```python
preprocessing.Operation(name: 'str', params: 'Dict[str, Any]' = <factory>, backend: 'str' = 'auto', requires_pixels: 'bool' = True, changes_geometry: 'bool' = False, terminal: 'bool' = False) -> None
```

Create or execute Operation for an image preprocessing pipeline.

**Parameters**

- `name` (str): Public name used in operation descriptions and traces.
- `params` (Dict[str, Any]; default `<factory>`): Operation parameters passed to the executor.
- `backend` (str; default `'auto'`): Requested pixel backend; auto selects a compatible implementation.
- `requires_pixels` (bool; default `True`): Whether execution needs decoded pixels.
- `changes_geometry` (bool; default `False`): Whether output coordinates differ from input coordinates.
- `terminal` (bool; default `False`): Whether execution yields a terminal value instead of an image.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.Operation("identity", {"function": lambda value: value})
print(operation(image).size)
```

Run [the standalone example](../../../examples/preprocessing/Operation.py) and see the [module index](../README.md).
