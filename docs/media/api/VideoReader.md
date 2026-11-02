# `media.VideoReader`

```python
media.VideoReader(path)
```

Read video frames sequentially or by random index through a single reusable OpenCV `VideoCapture` handle.

## Contract

Resolves `path` via `klygo.files.resolve()`, verifies the file exists, and opens `cv2.VideoCapture` once during initialization to record `.total_frames`, `.fps` (defaulting to `30.0` if unreported), `.width`, and `.height` before releasing the probe handle.

The active capture handle (`.cap`) is opened lazily on the first [`read_frame(index)`](#parameters) call:
- Tracks the current playhead position (`_current_pos`) so sequential reads (`0, 1, 2, ...`) call `cap.read()` directly without expensive `CAP_PROP_POS_FRAMES` seeks.
- Seeks automatically when `index != _current_pos`, and retries with an explicit seek if a read fails due to container desynchronization.
- Returns `None` cleanly for negative indices, indices `>= total_frames`, or unreadable frames instead of raising an exception.
- Call `.close()` or `.release()` to free the underlying OpenCV capture descriptor deterministically.

## Parameters

- `path`: `str | pathlib.Path` — path to an existing, readable video file.

## Returns

`VideoReader` — reader instance with attributes `.path`, `.total_frames`, `.fps`, `.width`, `.height`, and methods `.read_frame(index)`, `.release()`, and `.close()`.

## Errors and edge cases

Raises `FileNotFoundError` on initialization if `path` does not exist or cannot be opened by OpenCV `VideoCapture`. Out-of-range frame indices (`index < 0` or `index >= total_frames`) in `read_frame(index)` return `None`.

## Example

See [`VideoReader.py`](../../../examples/media/VideoReader.py) for a standalone runnable script.

## Complete executable example

```python
"""Read sequential and random-access BGR frames from a video using VideoReader."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [
        Image.new("RGB", (32, 24), "red"),
        Image.new("RGB", (32, 24), "green"),
        Image.new("RGB", (32, 24), "blue"),
    ]
    video = media.save_video(root / "clip.mp4", frames, fps=5.0)

    reader = media.VideoReader(video)
    try:
        assert reader.total_frames == 3
        assert (reader.width, reader.height) == (32, 24)

        first = reader.read_frame(0)
        last = reader.read_frame(2)
        assert first is not None and first.shape == (24, 32, 3)
        assert last is not None and last.shape == (24, 32, 3)
        assert reader.read_frame(99) is None
        print(first.shape)
    finally:
        reader.close()
```
