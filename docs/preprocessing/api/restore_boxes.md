# `preprocessing.restore_boxes`

Defined in `trace.py`.

```python
preprocessing.restore_boxes(boxes: 'Iterable[Sequence[float]]', trace: 'Iterable[Dict[str, Any]]') -> 'List[List[float]]'
```

Return or update restore boxes using preprocessing geometry history.

**Parameters**

- `boxes` (Iterable[Sequence[float]]): XYXY boxes to map back to source coordinates.
- `trace` (Iterable[Dict[str, Any]]): Recorded geometry transformations.

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
    print(p.restore_boxes([[4, 4, 20, 20]], history))
```

Run [the standalone example](../../../examples/preprocessing/restore_boxes.py) and see the [module index](../README.md).
