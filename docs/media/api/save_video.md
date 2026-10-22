# `media.save_video`

```python
media.save_video(output_path, frames, fps=30.0, fourcc='mp4v', overwrite=False, verbose=False)
```

Encode an iterable of images or lazy frames into a video file using OpenCV `VideoWriter`.

## Contract

Resolves `output_path` via `klygo.files.path()`, checks for existing files unless `overwrite=True`, and creates missing parent directories. Pulls the first frame from `frames` to determine the output `(width, height)` via [`media.to_array()`](to_array.md), initializes `cv2.VideoWriter` with `fourcc` and `fps`, and streams the remaining frames through the encoder one by one so bounded-memory iterators ([`MediaStream`](MediaStream.md) or generators) are encoded without buffering the entire video in RAM.

RGB inputs (`PIL.Image.Image`, [`LazyImage`](LazyImage.md), and 3-channel RGB arrays) are converted to BGR before writing; always releases the underlying `cv2.VideoWriter` in a `finally` block.

## Parameters

- `output_path`: `str | os.PathLike[str]` — destination video file path (e.g., `'clip.mp4'` or `'clip.avi'`).
- `frames`: `Iterable[PIL.Image.Image | numpy.ndarray | LazyImage]` — non-empty sequence or iterator of frames sharing the same dimensions.
- `fps`: `float`, default `30.0` — output playback frame rate in frames per second.
- `fourcc`: `str`, default `'mp4v'` — 4-character OpenCV codec code (such as `'mp4v'`, `'avc1'`, or `'MJPG'`).
- `overwrite`: `bool`, default `False` — allow replacing an existing file at `output_path`.
- `verbose`: `bool`, default `False` — display an encoding progress bar.

## Returns

`pathlib.Path` — resolved path to the encoded video file.

## Errors and edge cases

Raises `TypeError` for invalid argument types or unsupported frame objects, `FileExistsError` when `output_path` exists and `overwrite=False`, `ValueError` when `frames` is empty (`"frames iterable is empty"`), and `RuntimeError` if OpenCV cannot open the `VideoWriter` for the requested path and `fourcc`.

## Example

See [`save_video.py`](../../../examples/media/save_video.py) for a standalone runnable script.

## Complete executable example

```python
"""Encode a sequence of frames into an MP4 video and inspect its metadata."""

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

    video_path = media.save_video(root / "outputs" / "clip.mp4", frames, fps=5.0)
    info = media.probe(video_path)
    assert info["type"] == "video"
    assert info["frame_count"] == 3
    assert info["size"] == (32, 24)

    # Re-encode via MediaFrames.save_video() with overwrite=True
    loaded = media.load(video_path)
    loaded.save_video(video_path, fps=10.0, overwrite=True)
    loaded.close()
    print(info["frame_count"], info["size"])
```
