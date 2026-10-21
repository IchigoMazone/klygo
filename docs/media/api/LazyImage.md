# `media.LazyImage`

```python
media.LazyImage(path: Union[str, pathlib.Path, ForwardRef('LazyImage')], backend: str = 'pil', crop_box: Optional[Tuple[int, int, int, int]] = None, frame_index: Optional[int] = None, reader: Optional[klygo.media.operations.VideoReader] = None) -> None
```

Represent an image by path or frame reference and decode pixels on demand.

**Parameters:** Construct from a source image path or image object; optional backend and frame context tune loading.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing source when pixels are requested.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    lazy = media.LazyImage(source)
    print(lazy.size, lazy.is_loaded)
    print(lazy.to_array().shape)
```

Run [the standalone example](../../../examples/media/LazyImage.py). See [the module index](../README.md) for related APIs.
