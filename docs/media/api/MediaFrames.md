# `media.MediaFrames`

```python
media.MediaFrames(items: Optional[Iterable] = None, *, fps: float = 30.0, source_path: Union[str, pathlib.Path, NoneType] = None, width: Optional[int] = None, height: Optional[int] = None, source_type: str = 'video', reader: Optional[klygo.media.operations.VideoReader] = None) -> None
```

Hold indexable lazy image frames and source metadata.

**Parameters:** Construct from frame proxies; optional fps, source path, dimensions, and reader preserve media context.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** IndexError for an index outside the collection.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    frames = media.load(root)
    print(isinstance(frames, media.MediaFrames), frames.total_frames, frames.loaded_count)
```

Run [the standalone example](../../../examples/media/MediaFrames.py). See [the module index](../README.md) for related APIs.
