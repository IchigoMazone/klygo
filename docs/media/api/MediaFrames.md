# `media.MediaFrames`

```python
media.MediaFrames(items=None, *, fps=30.0, source_path=None, width=None, height=None, source_type='video', reader=None)
```

List-like container of [`LazyImage`](LazyImage.md) frames that preserves source metadata across slicing, mapping, and filtering.

## Contract

Subclasses Python's built-in `list` and is returned by [`media.load(..., stream=False)`](load.md). Stores frame proxies without decoding pixels until individual elements are materialized:
- **Indexing and slicing**: integer indexing (`frames[0]`) returns a single item; slicing (`frames[::2]`) returns a new `MediaFrames` retaining `source_path`, `width`, `height`, `source_type`, and `reader`, while scaling `fps` by `1 / abs(step)` when `abs(step) > 1`.
- **Functional & lazy methods**: `.map(func)` applies a callable to each item; `.transform(op_or_pipeline)` attaches a `klygo.preprocessing` `Operation` or `Compose` lazily to every frame; `.where(predicate)` (alias `.filter(predicate)`) keeps matching frames; `.copy()` clones the container without decoding pixels; `.to_list()` returns a plain `list`.
- **Persistence & lifecycle**: `.save_video(output_path, fps=None, **kwargs)` and `.save_images(output_dir, prefix='frame', **kwargs)` export all frames. `.unload()` clears cached RAM buffers across all items, and `.close()` (or using `with frames:`) unloads items and releases any shared [`VideoReader`](VideoReader.md).

## Parameters

- `items`: `Iterable[Any] | None`, default `None` — initial sequence of [`LazyImage`](LazyImage.md) or image objects.
- `fps`: `float`, default `30.0` — keyword-only; frame rate metadata (defaults to `30.0` if falsy).
- `source_path`: `str | pathlib.Path | None`, default `None` — keyword-only; origin path string (also accessible via `.source`).
- `width`: `int | None`, default `None` — keyword-only; frame width in pixels.
- `height`: `int | None`, default `None` — keyword-only; frame height in pixels.
- `source_type`: `str`, default `'video'` — keyword-only; `'image'`, `'folder'`, or `'video'`.
- `reader`: `VideoReader | None`, default `None` — keyword-only; shared video reader instance.

## Returns

`MediaFrames` — list-like frame collection exposing `.total_frames`, `.loaded_count`, `.source`, and `.fps`.

## Errors and edge cases

Raises `TypeError` if `.map()` or `.where()` / `.filter()` receives a non-callable, or if `.transform()` receives an object that is not a `klygo.preprocessing.Operation` or `Compose`.

## Example

See [`MediaFrames.py`](../../../examples/media/MediaFrames.py) for a standalone runnable script.

## Complete executable example

```python
"""Inspect, slice, filter, and unload a lazy MediaFrames collection."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    media.save(root / "1.png", Image.new("RGB", (32, 24), "red"))
    media.save(root / "2.png", Image.new("RGB", (64, 48), "blue"))

    with media.load(root) as frames:
        assert isinstance(frames, media.MediaFrames)
        assert frames.total_frames == 2 and frames.loaded_count == 0

        # Filter by zero-RAM header dimensions and slice
        large = frames.where(lambda img: img.width >= 64)
        assert len(large) == 1 and large[0].size == (64, 48)

        # Decode one frame, check loaded_count, then unload back to zero RAM
        _ = frames[0].load()
        assert frames.loaded_count == 1
        frames.unload()
        assert frames.loaded_count == 0
        print(isinstance(frames, media.MediaFrames), frames.total_frames, frames.loaded_count)
```
