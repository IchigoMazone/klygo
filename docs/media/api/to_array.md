# `media.to_array`

```python
media.to_array(image)
```

Convert a `LazyImage`, `PIL.Image.Image`, `numpy.ndarray`, or `torch.Tensor` into an independent `numpy.ndarray`.

## Contract

Normalizes diverse image representations into a standard NumPy array in `(H, W, C)` or `(H, W)` layout without aliasing the input buffer:
- [`LazyImage`](LazyImage.md): executes any queued PIL or `klygo.preprocessing` operations via `image.to_array()`.
- `numpy.ndarray`: returns an independent copy (`image.copy()`).
- `PIL.Image.Image`: converts via `np.array(image)`.
- `torch.Tensor` (when PyTorch is installed): detaches and moves the tensor to CPU, squeezes a leading batch dimension when `ndim == 4` and batch size is 1, permutes `CHW` (`shape[0] in (1, 3, 4)`) to `HWC`, and scales floating-point tensors in `[0.0, 1.0]` to `uint8` in `[0, 255]`.

## Parameters

- `image`: `LazyImage | PIL.Image.Image | numpy.ndarray | torch.Tensor` — input image representation to convert.

## Returns

`numpy.ndarray` — independent NumPy pixel array (typically `uint8` with shape `(H, W, C)` or `(H, W)`).

## Errors and edge cases

Raises `TypeError` when `image` is not a `LazyImage`, `PIL.Image.Image`, `numpy.ndarray`, or `torch.Tensor`. Propagates `FileNotFoundError` or `IndexError` if `image` is an unloaded `LazyImage` whose underlying file or video frame cannot be read.

## Example

See [`to_array.py`](../../../examples/media/to_array.py) for a standalone runnable script.

## Complete executable example

```python
"""Convert PIL images, LazyImage proxies, and NumPy arrays into independent arrays."""

import tempfile
import numpy as np
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    pil_img = Image.new("RGB", (32, 24), "red")

    # Convert from PIL Image
    arr = media.to_array(pil_img)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (24, 32, 3) and arr.dtype == np.uint8

    # Convert from LazyImage
    source = media.save(root / "source.png", pil_img)
    lazy_arr = media.to_array(media.LazyImage(source))
    assert np.array_equal(arr, lazy_arr)

    # Converting an existing ndarray returns an independent copy
    copied = media.to_array(arr)
    copied[0, 0, 0] = 0
    assert arr[0, 0, 0] == 255
    print(arr.shape, arr.dtype)
```
