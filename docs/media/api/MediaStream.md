# `media.MediaStream`

```python
media.MediaStream(source: Union[str, os.PathLike[str]], *, sample_rate: int = 1, max_frames: Optional[int] = None, recursive: bool = False, backend: str = 'pil', verbose: bool = False) -> None
```

Iterate frames from a source without retaining the whole decoded video.

**Parameters:** Construct from source plus sampling, recursion, backend, and frame-limit options.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing source; ValueError for invalid sampling.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    with media.load(root, stream=True) as stream:
        print(isinstance(stream, media.MediaStream), [frame.size for frame in stream])
```

Run [the standalone example](../../../examples/media/MediaStream.py). See [the module index](../README.md) for related APIs.
