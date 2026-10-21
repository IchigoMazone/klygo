# `media.probe`

```python
media.probe(path: Union[str, pathlib.Path], *, verbose: bool = False) -> Dict[str, Any]
```

Inspect image or video metadata without loading a frame collection.

**Parameters:**
- `path`: existing media file.
- `verbose`: print the metadata as a Rich table while returning the dictionary (default `False`).

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileNotFoundError for missing file; ValueError for a directory.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    source = media.save(root / "source.png", image)
    print(media.probe(source)["type"], media.probe(source)["size"])
```

Run [the standalone example](../../../examples/media/probe.py). See [the module index](../README.md) for related APIs.
