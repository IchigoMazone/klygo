# `media.LazyImage`

```python
media.LazyImage(path, backend='pil', crop_box=None, frame_index=None, reader=None)
```

Represent an image file or video frame as a zero-RAM proxy that defers pixel decoding and preprocessing until materialized.

## Contract

Subclasses `PIL.Image.Image` and implements `os.PathLike` (`__fspath__`) and the NumPy array protocol (`__array__`):
- **Zero-RAM metadata**: `.size`, `.width`, `.height`, `.shape`, `.dtype`, and `.mode` read image headers or video metadata without decoding pixel buffers (`is_loaded` remains `False`).
- **Zero-RAM cropping**: `.lazy_crop((xmin, ymin, xmax, ymax))` and `.crop(box)` (unless `lazy=False` is passed) return a new `LazyImage` that composes relative crop offsets in memory without reading pixels.
- **Lazy PIL & preprocessing pipelines**: calling `.process(op_or_pipeline)` attaches `klygo.preprocessing` operations, and calling non-destructive PIL methods (`resize`, `convert`, `rotate`, `transpose`, `filter`, `point`, `quantize`, `transform`) queues instructions on a cloned `LazyImage`.
- **Materialization & cache control**: `.load(cache=True)` decodes raw source pixels; `.to_pil(cache=False)`, `.to_array(cache=False)`, and `.to_tensor(cache=False)` decode and execute all queued operations, recording execution metadata in `.trace`. Call `.unload()` (or `.clear()`) to free RAM.

## Parameters

- `path`: `str | pathlib.Path | LazyImage` — source image/video file path or an existing `LazyImage` to clone.
- `backend`: `str`, default `'pil'` — `'pil'` (RGB `PIL.Image.Image`) or `'opencv'` (BGR `numpy.ndarray`).
- `crop_box`: `tuple[int, int, int, int] | None`, default `None` — `(xmin, ymin, xmax, ymax)` region to crop upon decoding.
- `frame_index`: `int | None`, default `None` — 0-based video frame index when backed by a video file.
- `reader`: `VideoReader | None`, default `None` — optional shared [`VideoReader`](VideoReader.md) for video frame decoding.

## Returns

`LazyImage` — lazy image proxy exposing `.path`, `.url`, `.size`, `.shape`, `.is_loaded`, `.trace`, `.to_pil()`, `.to_array()`, and `.to_tensor()`.

## Errors and edge cases

Raises `FileNotFoundError` when `.load()` cannot find or read the image/video file, `IndexError` when `frame_index` cannot be decoded from the video, `ImportError` if `.to_tensor()` is called without PyTorch installed, and `TypeError` when assigning an unsupported object to `.image`.

## Example

See [`LazyImage.py`](../../../examples/media/LazyImage.py) for a standalone runnable script.

## Complete executable example

```python
"""Inspect, crop, and materialize a LazyImage without premature RAM allocation."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    source = media.save(root / "source.png", Image.new("RGB", (32, 24), "red"))

    lazy = media.LazyImage(source)
    assert lazy.size == (32, 24) and not lazy.is_loaded

    # Zero-RAM nested crop
    cropped = lazy.lazy_crop((4, 4, 20, 16))
    assert cropped.size == (16, 12) and not cropped.is_loaded

    # Materialize to NumPy array on demand
    arr = cropped.to_array()
    assert arr.shape == (12, 16, 3)
    print(lazy.size, lazy.is_loaded, arr.shape)
```
