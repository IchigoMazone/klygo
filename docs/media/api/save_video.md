# `media.save_video`

```python
media.save_video(output_path: Union[str, pathlib.Path], frames: Iterable[Union[PIL.Image.Image, numpy.ndarray]], fps: float = 30.0, fourcc: str = 'mp4v', overwrite: bool = False, verbose: bool = False) -> pathlib.Path
```

Encode an iterable of equally sized frames as a video.

**Parameters:** output_path is a video path; frames is an iterable; fps and fourcc control encoding.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** ValueError for empty frames; RuntimeError if the video writer cannot open.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
    video = media.save_video(root / "clip.mp4", frames, fps=5)
    print(media.probe(video)["frame_count"])
```

Run [the standalone example](../../../examples/media/save_video.py). See [the module index](../README.md) for related APIs.
