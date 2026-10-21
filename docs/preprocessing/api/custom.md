# `preprocessing.custom`

Defined in `core.py`.

```python
preprocessing.custom(function: 'Callable[[Any], Any]', backend: 'str' = 'auto', name: 'Optional[str]' = None, **kwargs: 'Any') -> 'Operation'
```

Create or execute custom for an image preprocessing pipeline.

**Parameters**

- `function` (Callable[[Any], Any]): Callable transformation applied to image data.
- `backend` (str; default `'auto'`): Requested pixel backend; auto selects a compatible implementation.
- `name` (Optional[str]; default `None`): Public name used in operation descriptions and traces.
- `kwargs` (Any): Value passed to the operation or evaluator.

**Returns and effects:** operation, pipeline, predicate, metric, or evaluated image value. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Non-callable executors, unsupported pipeline items, or invalid inputs raise the relevant `TypeError` or `ValueError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
operation = p.custom(lambda value: value.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
print(operation(image).size)
```

Run [the standalone example](../../../examples/preprocessing/custom.py) and see the [module index](../README.md).
