# `media.to_tensor`

```python
media.to_tensor(image: Any, normalize: bool = True) -> Any
```

Convert image data to a CHW PyTorch tensor.

**Parameters:** image is an accepted image representation; normalize scales uint8 pixels to float values in 0–1.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** ImportError if PyTorch is unavailable; TypeError for unsupported input.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    tensor = media.to_tensor(Image.new("RGB", (32, 24), "red"), normalize=True)
    print(tuple(tensor.shape), str(tensor.dtype))
```

Run [the standalone example](../../../examples/media/to_tensor.py). See [the module index](../README.md) for related APIs.
