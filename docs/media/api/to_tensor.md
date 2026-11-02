# `media.to_tensor`

```python
media.to_tensor(image, normalize=True)
```

Convert a `LazyImage`, `PIL.Image.Image`, `numpy.ndarray`, or `torch.Tensor` into a `CHW` PyTorch tensor.

## Contract

Requires the optional `torch` dependency (`pip install "klygo[torch]"`). Converts supported image inputs into a 3D channel-first (`C, H, W`) `torch.Tensor`:
- [`LazyImage`](LazyImage.md): materializes the lazy pipeline via `image.to_tensor(cache=False)`.
- `PIL.Image.Image` or `numpy.ndarray`: converts 2D grayscale `(H, W)` inputs to `(1, H, W)` and permutes 3D `(H, W, C)` inputs to `(C, H, W)`.
- `torch.Tensor`: detaches and clones the tensor (`image.detach().clone()`) so the caller receives an independent tensor.
- When `normalize=True` and the resulting tensor has `dtype == torch.uint8`, converts it to `torch.float32` and divides by `255.0` to produce values in `[0.0, 1.0]`.

## Parameters

- `image`: `LazyImage | PIL.Image.Image | numpy.ndarray | torch.Tensor` — input image representation to convert.
- `normalize`: `bool`, default `True` — scale `uint8` pixel values `[0, 255]` to `float32` values in `[0.0, 1.0]`.

## Returns

`torch.Tensor` — independent channel-first `(C, H, W)` tensor (`torch.float32` when normalized from `uint8`, or `torch.uint8` when `normalize=False`).

## Errors and edge cases

Raises `ImportError` if PyTorch is not installed. Raises `TypeError` if `image` is not one of the supported image representations.

## Example

See [`to_tensor.py`](../../../examples/media/to_tensor.py) for a standalone runnable script.

## Complete executable example

```python
"""Convert PIL images and NumPy arrays into normalized or raw CHW PyTorch tensors."""

import numpy as np
from PIL import Image
from klygo import media

pil_img = Image.new("RGB", (32, 24), "red")

# Normalized float32 tensor in [0.0, 1.0] with shape (C, H, W)
tensor_norm = media.to_tensor(pil_img, normalize=True)
assert tuple(tensor_norm.shape) == (3, 24, 32)
assert str(tensor_norm.dtype) == "torch.float32"
assert float(tensor_norm[0, 0, 0]) == 1.0

# Unnormalized uint8 tensor in [0, 255]
tensor_raw = media.to_tensor(pil_img, normalize=False)
assert tuple(tensor_raw.shape) == (3, 24, 32)
assert str(tensor_raw.dtype) == "torch.uint8"
assert int(tensor_raw[0, 0, 0]) == 255

# 2D grayscale array expands to 1 channel (1, H, W)
gray = np.full((24, 32), 128, dtype=np.uint8)
assert tuple(media.to_tensor(gray).shape) == (1, 24, 32)
print(tuple(tensor_norm.shape), str(tensor_norm.dtype))
```
