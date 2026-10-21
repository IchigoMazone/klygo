# `media.convert`

```python
media.convert(source: Union[str, pathlib.Path], target: Union[str, pathlib.Path], codec: Optional[str] = None, crf: int = 23, fps: Optional[float] = None, gpu: Optional[bool] = None, preset: Optional[str] = None, overwrite: bool = False, verbose: bool = False) -> pathlib.Path
```

Convert one image format to another, or transcode a video.

**Parameters:** source and target determine media formats; codec, crf, fps, gpu, and preset tune video conversion.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing source; ValueError for incompatible media kinds.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    target = media.convert(source, root / "converted.jpg")
    print(media.probe(target)["format"])
```

Run [the standalone example](../../../examples/media/convert.py). See [the module index](../README.md) for related APIs.
