# `media.save`

```python
media.save(path: Union[str, pathlib.Path], image: Union[PIL.Image.Image, numpy.ndarray], overwrite: bool = False, verbose: bool = False) -> pathlib.Path
```

Write a PIL or NumPy image to a file.

**Parameters:** path is the destination; image is a PIL image, array, or LazyImage; overwrite permits replacement.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** FileExistsError when the destination exists; TypeError for unsupported image objects.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    destination = media.save(root / "image.png", Image.new("RGB", (32, 24), "red"))
    print(media.probe(destination)["size"])
```

Run [the standalone example](../../../examples/media/save.py). See [the module index](../README.md) for related APIs.
