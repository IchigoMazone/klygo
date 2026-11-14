# `media.info`

```python
media.info(path)
```

Read header metadata from an image or video file without decoding full frame streams into memory.

## Contract

Resolves `path` via `klygo.files.path()` and verifies that it points to an existing regular file:
- **For video files** (`.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`): opens `cv2.VideoCapture` briefly to inspect container properties, normalizes FourCC codes into canonical codec names (`'h264'`, `'h265'`, `'mp4v'`, `'mjpeg'`, `'vp9'`, `'vp8'`, `'av1'`), and computes `web_ready` (`True` when the container is `.mp4` or `.webm` with an H.264/VP8/VP9/AV1 codec) and `duration_seconds`.
- **For image files**: opens the file header with `PIL.Image.open()` to read `format`, `mode`, `width`, `height`, and `size` without decoding compressed pixel blocks.

## Parameters

- `path`: `str | os.PathLike[str]` — path to an existing image or video file.

## Returns

`dict[str, Any]` — metadata dictionary:
- Image keys: `'name'`, `'path'`, `'type'` (`'image'`), `'format'`, `'mode'`, `'width'`, `'height'`, `'size'` (`(width, height)`).
- Video keys: `'name'`, `'path'`, `'type'` (`'video'`), `'format'`, `'codec'`, `'web_ready'`, `'width'`, `'height'`, `'size'`, `'fps'`, `'frame_count'`, `'duration_seconds'`.

## Errors and edge cases

Raises `FileNotFoundError` when `path` does not exist or a video cannot be opened by OpenCV. Raises `ValueError` if `path` points to a directory rather than a regular file. Unrecognized image formats propagate `PIL.UnidentifiedImageError`.

## Example

See [`info.py`](../../../examples/media/info.py) for a standalone runnable script.

## Complete executable example

```python
"""Inspect metadata of both image and video files."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    img_path = media.save(root / "sample.png", Image.new("RGB", (64, 48), "navy"))
    vid_path = root / "clip.mp4"
    media.save_video(vid_path, [Image.new("RGB", (64, 48), "navy")] * 5, fps=5.0)

    img_meta = media.info(img_path)
    assert img_meta["type"] == "image"
    assert img_meta["format"] == "PNG"
    assert img_meta["size"] == (64, 48)

    vid_meta = media.info(vid_path)
    assert vid_meta["type"] == "video"
    assert vid_meta["frame_count"] == 5
    assert vid_meta["duration_seconds"] == 1.0
    print(img_meta["type"], img_meta["size"], vid_meta["duration_seconds"])
```
