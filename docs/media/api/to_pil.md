# `media.to_pil`

```python
media.to_pil(image: Any) -> PIL.Image.Image
```

Convert an array, tensor, or lazy image to a PIL image.

**Parameters:** image is a PIL, NumPy, Torch, or LazyImage image.

**Returns and effects:** See the signature above; file operations write to the requested destination, lazy containers defer pixel decoding, and converters return a new image representation.

**Errors:** TypeError for unsupported data; ValueError for unsupported array shape.

```python
import tempfile
from PIL import Image
from klygo import media, files
with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    import numpy as np
    image = media.to_pil(np.zeros((24, 32, 3), dtype=np.uint8))
    print(image.size, image.mode)
```

Run [the standalone example](../../../examples/media/to_pil.py). See [the module index](../README.md) for related APIs.
