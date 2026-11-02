# `preprocessing.original_size`

Defined in `trace.py`.

```python
preprocessing.original_size(image: 'Any') -> 'Tuple[int, int]'
```

Return or update original size using preprocessing geometry history.

**Parameters**

- `image` (Any): Input PIL, NumPy, or lazy image.

**Returns and effects:** trace data, restored geometry, or updated image. Image operations remain deferred when applied to a `media.LazyImage`; calling `materialize` decodes pixels.

**Errors:** Malformed image data, coordinates, or trace entries can raise their underlying `TypeError`, `ValueError`, or `KeyError`.

```python
from PIL import Image
from klygo import preprocessing as p, media
image = Image.new('RGB', (64, 48), (120, 80, 40))
import tempfile
from klygo import files
with tempfile.TemporaryDirectory() as temporary:
    source = media.save(files.path(temporary) / "input.png", image)
    lazy = media.load(source)[0]
    processed = p.resize((32, 24))(lazy)
    p.materialize(processed, output="pil")
    history = p.get_trace(processed)
    print(p.original_size(processed))
```

Run [the standalone example](../../../examples/preprocessing/original_size.py) and see the [module index](../README.md).
