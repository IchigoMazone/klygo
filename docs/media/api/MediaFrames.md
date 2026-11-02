# `media.MediaFrames`

```python
media.MediaFrames(source, *, stream=False, indexed=False, sample_rate=1, max_frames=None, recursive=False, backend='pil', verbose=False)
```

Unified container for indexed lazy frames and bounded-memory sequential streaming.

## Contract

[`media.load()`](load.md) always returns `MediaFrames`:

- With `stream=False`, the container is list-like. It supports indexing, slicing, `.loaded_count`, `.to_list()`, mutation, and `.unload()` while keeping pixels lazy until accessed.
- With `indexed=True`, its path/frame-number index remains materialized while `LazyImage` objects are created only when accessed. Live proxies are weakly cached, explicit replacements remain authoritative, and slicing preserves indexed storage.
- With `stream=True`, the container reads one [`LazyImage`](LazyImage.md) at a time. Random access and list mutation raise `TypeError`, while `.sample()`, `.limit()`, `.where()`, quality filtering, motion filtering, deduplication, and preprocessing remain lazy.
- All modes expose `.is_stream`, `.is_indexed`, `.source`, `.source_type`, `.total_frames`, `.fps`, `.width`, and `.height`.
- `.transform(callable)` applies a collection-level callable once. Use `.map(callable)` for a frame-level transform.
- `.save_video()` and `.save_images()` consume either mode sequentially.

Calling `MediaFrames(source, stream=False)` delegates to `media.load(source, stream=False)`. Calling it with `stream=True` creates the bounded-memory mode directly.

## Parameters

- `source`: `str | os.PathLike[str]` — existing image, video, or image directory.
- `stream`: `bool`, default `False` — keyword-only; select bounded-memory sequential mode when true.
- `indexed`: `bool`, default `False` — keyword-only; use path-backed random access without eager proxy allocation.
- `sample_rate`: `int`, default `1` — emit every nth frame in streaming mode.
- `max_frames`: `int | None`, default `None` — maximum frames emitted in streaming mode.
- `recursive`: `bool`, default `False` — search image directories recursively.
- `backend`: `str`, default `'pil'` — pixel backend, either `'pil'` or `'opencv'`.
- `verbose`: `bool`, default `False` — enable progress output where supported.

## Returns

`MediaFrames` — the same public type in indexed and streaming modes.

## Errors and edge cases

Raises `TypeError` for invalid parameters or when an indexed-only operation is used in streaming mode. Raises `FileNotFoundError` when the source does not exist. Raises `ValueError` for unsupported sources, invalid backends, or non-positive sampling and limit values.

## Example

See [`MediaFrames.py`](../../../examples/media/MediaFrames.py) for a standalone runnable script.

## Complete executable example

```python
"""Use MediaFrames in indexed and bounded-memory modes."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    media.save(root / "1.png", Image.new("RGB", (32, 24), "red"))
    media.save(root / "2.png", Image.new("RGB", (64, 48), "blue"))

    indexed = media.load(root, stream=False)
    compact_index = media.load(root, indexed=True)
    streamed = media.load(root, stream=True, max_frames=1)

    assert isinstance(indexed, media.MediaFrames)
    assert isinstance(streamed, media.MediaFrames)
    assert not indexed.is_stream and streamed.is_stream
    assert compact_index.is_indexed and compact_index[0] is compact_index[0]
    assert indexed[0].size == (32, 24)
    assert len(list(streamed)) == 1
```
