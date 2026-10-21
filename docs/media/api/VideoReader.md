# `media.VideoReader`

```python
media.VideoReader(path: Union[str, pathlib.Path]) -> None
```

Reuse one OpenCV capture for sequential and random video-frame access.

**Parameters:** path is an existing video file.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing path or unreadable video.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    video = media.save_video(root / "clip.mp4", [Image.new("RGB", (32, 24), "red")], fps=5)
    reader = media.VideoReader(video)
    try:
        print(reader.read_frame(0).shape)
    finally:
        reader.close()
```

Run [the standalone example](../../../examples/media/VideoReader.py). See [the module index](../README.md) for related APIs.
