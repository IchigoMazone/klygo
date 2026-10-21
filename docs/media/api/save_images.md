# `media.save_images`

```python
media.save_images(output_dir: Union[str, pathlib.Path], images: Iterable[Union[PIL.Image.Image, numpy.ndarray]], prefix: str = 'frame', extension: str = '.jpg', overwrite: bool = False, verbose: bool = False) -> List[pathlib.Path]
```

Write a sequence of numbered image files.

**Parameters:** output_dir receives generated filenames; prefix and extension customize them; overwrite controls collisions.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileExistsError when a generated name exists without overwrite.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
    paths = media.save_images(root / "frames", frames, extension=".png")
    print([files.name(path) for path in paths])
```

Run [the standalone example](../../../examples/media/save_images.py). See [the module index](../README.md) for related APIs.
