# `media.iter_frames`

```python
media.iter_frames(source: Union[str, pathlib.Path], sample_rate: int = 1, recursive: bool = False, backend: str = 'pil', verbose: bool = False) -> Generator[Union[PIL.Image.Image, numpy.ndarray], NoneType, NoneType]
```

Yield sampled frames from a video or image directory.

**Parameters:** source is an image directory or video; sample_rate takes every nth frame; backend selects PIL or OpenCV.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** ValueError for sample_rate below one; FileNotFoundError for missing input.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    print([frame.size for frame in media.iter_frames(root, sample_rate=1)])
```

Run [the standalone example](../../../examples/media/iter_frames.py). See [the module index](../README.md) for related APIs.
