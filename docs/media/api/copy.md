# `media.copy`

```python
media.copy(source: Union[str, pathlib.Path], target: Union[str, pathlib.Path], overwrite: bool = False) -> pathlib.Path
```

Copy a media file or directory.

**Parameters:** source is an existing path; target is the new path; overwrite controls replacement.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing source; FileExistsError if target exists without overwrite.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    duplicate = media.copy(source, root / "copy.png")
    print(media.probe(duplicate)["size"])
```

Run [the standalone example](../../../examples/media/copy.py). See [the module index](../README.md) for related APIs.
