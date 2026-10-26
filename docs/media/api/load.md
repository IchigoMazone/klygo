# `media.load`

```python
media.load(source, recursive=False, backend='pil', verbose=False, *, stream=False, sample_rate=1, max_frames=None)
```

Load an image file, video file, or image directory as indexed lazy frames or a bounded-memory sequential stream.

## Contract

Resolves `source` via `klygo.files.path()`. When `stream=False` (the default), inspects the source without decoding full pixel buffers into RAM and returns a [`MediaFrames`](MediaFrames.md) list-like container of [`LazyImage`](LazyImage.md) proxies:
- For a single supported image (`.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp`), returns a 1-element `MediaFrames` with `source_type="image"`.
- For a directory, discovers matching image files in case-insensitive sorted path order (recursively when `recursive=True`) and returns a `MediaFrames` with `source_type="folder"`.
- For a supported video (`.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`), probes header metadata once via [`VideoReader`](VideoReader.md) and populates `LazyImage` proxies indexed from `0` to `total_frames - 1` sharing that reader with `source_type="video"`.

When `stream=True`, the same [`MediaFrames`](MediaFrames.md) API yields frames sequentially with bounded memory according to `sample_rate` and `max_frames`. Passing `sample_rate != 1` or `max_frames is not None` while `stream=False` is rejected.

## Parameters

- `source`: `str | os.PathLike[str]` — path to an existing image file, video file, or directory containing images.
- `recursive`: `bool`, default `False` — traverse subdirectories when `source` is a directory.
- `backend`: `str`, default `'pil'` — decoded pixel representation (`'pil'` for RGB `PIL.Image.Image`, `'opencv'` for BGR `numpy.ndarray`).
- `verbose`: `bool`, default `False` — enable progress reporting where supported.
- `stream`: `bool`, default `False` — keyword-only; select sequential mode instead of indexed mode.
- `sample_rate`: `int`, default `1` — keyword-only; emit every `sample_rate`-th frame when `stream=True`.
- `max_frames`: `int | None`, default `None` — keyword-only; cap the maximum number of yielded frames when `stream=True`.

## Returns

`MediaFrames` — indexed lazy frame collection when `stream=False`, or bounded-memory iterable when `stream=True`.

## Errors and edge cases

Raises `TypeError` if any parameter has an invalid type. Raises `FileNotFoundError` when `source` does not exist or a video cannot be opened. Raises `ValueError` if `backend` is not `'pil'` or `'opencv'`, if `source` has an unsupported file extension or is an empty image directory, or if `sample_rate != 1` or `max_frames` is provided without `stream=True`.

## Example

See [`load.py`](../../../examples/media/load.py) for a standalone runnable script.

## Complete executable example

```python
"""Load images and directories in indexed or streaming MediaFrames mode."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    first = media.save(root / "a.png", Image.new("RGB", (32, 24), "red"))
    second = media.save(root / "b.png", Image.new("RGB", (32, 24), "blue"))

    # Indexed lazy collection (Zero-RAM until pixel access)
    frames = media.load(root)
    assert isinstance(frames, media.MediaFrames)
    assert len(frames) == 2 and frames.loaded_count == 0
    assert frames[0].size == (32, 24)

    # Sequential bounded-memory stream with frame sampling
    with media.load(root, stream=True, sample_rate=1, max_frames=1) as stream:
        assert isinstance(stream, media.MediaFrames)
        sampled = list(stream)
        assert len(sampled) == 1 and sampled[0].size == (32, 24)
    print(len(frames), frames[0].size, type(stream).__name__)
```
