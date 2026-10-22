# `media.save`

```python
media.save(path, image, overwrite=False, verbose=False)
```

Save a single image to disk at the requested path, creating parent directories automatically.

## Contract

Resolves `path` via `klygo.files.path()` and creates any missing parent directories via `klygo.files.mkdir()`. Refuses to overwrite an existing file unless `overwrite=True`. Accepts three image representations:
- [`LazyImage`](LazyImage.md): materializes the image and any attached preprocessing operations as a `PIL.Image.Image` without permanently caching the buffer (`image.to_pil(cache=False)`) before saving.
- `PIL.Image.Image`: writes directly using Pillow's `image.save(p)`.
- `numpy.ndarray`: encodes via OpenCV `cv2.imwrite(str(p), image)` (expects BGR channel order for 3-channel color arrays).

## Parameters

- `path`: `str | os.PathLike[str]` — destination file path whose extension determines the encoded image format.
- `image`: `PIL.Image.Image | numpy.ndarray | LazyImage` — pixel data or lazy image proxy to write.
- `overwrite`: `bool`, default `False` — allow replacing an existing file at `path`.
- `verbose`: `bool`, default `False` — display a progress bar while saving.

## Returns

`pathlib.Path` — resolved path to the written image file.

## Errors and edge cases

Raises `TypeError` when `path`, `overwrite`, or `verbose` has an invalid type, or when `image` is not a `PIL.Image.Image`, `numpy.ndarray`, or `LazyImage`. Raises `FileExistsError` if `path` already exists and `overwrite=False`. Raises `ValueError` if OpenCV fails to encode a `numpy.ndarray` to the target extension.

## Example

See [`save.py`](../../../examples/media/save.py) for a standalone runnable script.

## Complete executable example

```python
"""Save PIL images, LazyImage proxies, and NumPy arrays to disk."""

import tempfile
import numpy as np
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)

    # Save a PIL Image into a nested directory
    pil_path = media.save(root / "nested" / "red.png", Image.new("RGB", (32, 24), "red"))
    assert files.is_file(pil_path)
    assert media.probe(pil_path)["size"] == (32, 24)

    # Save a cropped LazyImage and replace an existing file with overwrite=True
    lazy_crop = media.LazyImage(pil_path).lazy_crop((0, 0, 16, 12))
    media.save(pil_path, lazy_crop, overwrite=True)
    assert media.probe(pil_path)["size"] == (16, 12)

    # Save a NumPy array
    arr_path = media.save(root / "array.png", np.zeros((20, 30, 3), dtype=np.uint8))
    print(media.probe(pil_path)["size"], media.probe(arr_path)["size"])
```
