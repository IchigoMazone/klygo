# `media.load`

```python
media.load(source: Union[str, os.PathLike[str]], recursive: bool = False, backend: str = 'pil', verbose: bool = False, *, stream: bool = False, sample_rate: int = 1, max_frames: Optional[int] = None) -> Union[klygo.media.operations.MediaFrames, klygo.media.operations.MediaStream]
```

Load image, video, or directory content as lazy frames. Set stream=True to consume frames sequentially.

**Parameters:** source is an existing image, video, or directory path; stream, sample_rate, and max_frames control iteration.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing input; ValueError for invalid backend or stream-only options.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    frames = media.load(source)
    print(len(frames), frames[0].size)
    print(type(media.load(source, stream=True)).__name__)
```

Run [the standalone example](../../../examples/media/load.py). See [the module index](../README.md) for related APIs.
