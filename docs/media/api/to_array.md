# `media.to_array`

```python
media.to_array(image: Any) -> numpy.ndarray
```

Convert PIL, NumPy, LazyImage, or Torch image data into a NumPy array.

**Parameters:** image is an accepted image representation.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** TypeError for unsupported input.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    array = media.to_array(Image.new("RGB", (32, 24), "red"))
    print(array.shape, array.dtype)
```

Run [the standalone example](../../../examples/media/to_array.py). See [the module index](../README.md) for related APIs.
