# `media.MediaStream`

```python
media.MediaStream(source, *, sample_rate=1, max_frames=None, recursive=False, backend='pil', verbose=False)
```

Yield [`LazyImage`](LazyImage.md) frames sequentially from a video, image, or directory with bounded memory and lazy filtering.

## Contract

Returned by [`media.load(..., stream=True)`](load.md) or constructed directly. Validates `source` and probes header metadata on initialization (setting `.source_type`, `.width`, `.height`, `.fps`, and `.total_frames`), but opens a [`VideoReader`](VideoReader.md) or directory iterator only when iterated (`for frame in stream:`).

Supports immutable, clone-on-write pipeline chaining before iteration:
- `.transform(op_or_pipeline)`: attaches a `klygo.preprocessing.Operation` or `Compose` to every emitted frame.
- `.where(predicate)`: keeps only frames where `predicate(frame)` is truthy (sets `total_frames = None`).
- `.where_quality(*, min_brightness=None, max_brightness=None, min_contrast=None, min_sharpness=None, max_blur=None, min_entropy=None, preview_size=(160, 160))`: filters frames using `klygo.preprocessing` quality scores.
- `.where_motion(threshold=0.02)`: drops frames whose motion score relative to the previous kept frame is below `threshold`.
- `.deduplicate(threshold=0.95)`: drops frames whose similarity score relative to the previous kept frame is at least `threshold`.

## Parameters

- `source`: `str | os.PathLike[str]` — existing video file, image file, or image directory.
- `sample_rate`: `int`, default `1` — keyword-only; emit every `sample_rate`-th source frame (`>= 1`).
- `max_frames`: `int | None`, default `None` — keyword-only; stop after yielding `max_frames` frames (`>= 1`).
- `recursive`: `bool`, default `False` — keyword-only; search subdirectories when `source` is a folder.
- `backend`: `str`, default `'pil'` — keyword-only; `'pil'` or `'opencv'`.
- `verbose`: `bool`, default `False` — keyword-only; store verbosity flag on the stream.

## Returns

`MediaStream` — reusable context-managed iterable yielding [`LazyImage`](LazyImage.md) instances.

## Errors and edge cases

Raises `TypeError` for invalid argument types or non-callable predicates/transforms, `FileNotFoundError` when `source` does not exist, and `ValueError` when `sample_rate < 1`, `max_frames < 1`, `backend` is invalid, or `source` is an unsupported file or empty folder.

## Example

See [`MediaStream.py`](../../../examples/media/MediaStream.py) for a standalone runnable script.

## Complete executable example

```python
"""Stream frames with sampling, predicates, and automatic resource cleanup."""

import tempfile
from PIL import Image
from klygo import files, media, preprocessing

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    for idx, color in enumerate(("red", "green", "blue", "yellow")):
        media.save(root / f"{idx}.png", Image.new("RGB", (32, 24), color))

    with media.MediaStream(root, sample_rate=2, max_frames=2) as stream:
        resized_stream = stream.transform(preprocessing.resize((16, 12)))
        frames = list(resized_stream)
        assert len(frames) == 2
        assert [f.to_pil().size for f in frames] == [(16, 12), (16, 12)]
        print(isinstance(stream, media.MediaStream), [f.size for f in stream])
```
