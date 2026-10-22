# `media.to_pil`

```python
media.to_pil(image)
```

Convert a `LazyImage`, `PIL.Image.Image`, `numpy.ndarray`, or `torch.Tensor` into an independent `PIL.Image.Image`.

## Contract

Produces an independent `PIL.Image.Image` instance from any supported in-memory or lazy image representation:
- [`LazyImage`](LazyImage.md): materializes the image and any attached preprocessing operations via `image.to_pil()`.
- `PIL.Image.Image`: returns a defensive copy (`image.copy()`).
- `numpy.ndarray` or `torch.Tensor`: first converts to a NumPy array via [`media.to_array()`](to_array.md), then maps 2D `(H, W)` arrays and 3D `(H, W, 1)` single-channel arrays to grayscale `PIL.Image.Image`, `(H, W, 3)` arrays to `'RGB'`, and `(H, W, 4)` arrays to `'RGBA'`.

## Parameters

- `image`: `LazyImage | PIL.Image.Image | numpy.ndarray | torch.Tensor` — input image representation to convert.

## Returns

`PIL.Image.Image` — independent Pillow image instance.

## Errors and edge cases

Raises `TypeError` if `image` is not a supported image representation. Raises `ValueError` if the intermediate array does not have 2 dimensions or 3 dimensions with 1, 3, or 4 channels.

## Example

See [`to_pil.py`](../../../examples/media/to_pil.py) for a standalone runnable script.

## Complete executable example

```python
"""Convert NumPy arrays and LazyImage instances into independent PIL Images."""

import tempfile
import numpy as np
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)

    # Convert 3-channel RGB and 4-channel RGBA NumPy arrays
    rgb_arr = np.zeros((24, 32, 3), dtype=np.uint8)
    pil_rgb = media.to_pil(rgb_arr)
    assert isinstance(pil_rgb, Image.Image)
    assert pil_rgb.size == (32, 24) and pil_rgb.mode == "RGB"

    rgba_arr = np.zeros((24, 32, 4), dtype=np.uint8)
    pil_rgba = media.to_pil(rgba_arr)
    assert pil_rgba.mode == "RGBA"

    # Convert a LazyImage proxy
    saved = media.save(root / "sample.png", pil_rgb)
    from_lazy = media.to_pil(media.LazyImage(saved))
    assert from_lazy.size == (32, 24)
    print(pil_rgb.size, pil_rgb.mode)
```
