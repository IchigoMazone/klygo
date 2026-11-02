# `media.iter_frames`

```python
media.iter_frames(source, sample_rate=1, recursive=False, backend='pil', verbose=False)
```

Yield sampled frames from a video file, single image, or image directory as a generator.

## Contract

Resolves `source` via `klygo.files.path()` and validates `sample_rate >= 1` and `backend in ('pil', 'opencv')`:
- When `source` is a video file (`.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`), opens a `cv2.VideoCapture` stream, reads sequentially, and yields every `sample_rate`-th decoded frame (`PIL.Image.Image` in RGB mode when `backend='pil'`, or BGR `numpy.ndarray` when `backend='opencv'`), closing the capture handle automatically in a `finally` block when iteration finishes or breaks early.
- When `source` is an image file or directory, loads [`MediaFrames`](MediaFrames.md) via [`media.load()`](load.md) and yields every `sample_rate`-th [`LazyImage`](LazyImage.md).

## Parameters

- `source`: `str | os.PathLike[str]` — existing video file, image file, or directory of images.
- `sample_rate`: `int`, default `1` — step interval; yields frames where `frame_index % sample_rate == 0`.
- `recursive`: `bool`, default `False` — traverse subdirectories when `source` is a directory.
- `backend`: `str`, default `'pil'` — `'pil'` for RGB `PIL.Image.Image` / `LazyImage` or `'opencv'` for BGR `numpy.ndarray`.
- `verbose`: `bool`, default `False` — display a frame iteration progress bar.

## Returns

`Generator[PIL.Image.Image | numpy.ndarray | LazyImage, None, None]` — lazy generator yielding sampled frames on demand.

## Errors and edge cases

Raises `ValueError` when `sample_rate < 1`, when `backend` is not `'pil'` or `'opencv'`, or when `source` is an unsupported file or empty directory. Raises `FileNotFoundError` when `source` does not exist or the video stream cannot be opened. Note that because `iter_frames` is a generator function, validation runs when the first item is requested (`next()` or `for` loop).

## Example

See [`iter_frames.py`](../../../examples/media/iter_frames.py) for a standalone runnable script.

## Complete executable example

```python
"""Iterate sampled frames from an image directory and a video file."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "green", "blue", "yellow")]
    media.save_images(root / "imgs", frames, extension=".png")

    # Yield every 2nd image from a directory
    sampled_imgs = list(media.iter_frames(root / "imgs", sample_rate=2))
    assert len(sampled_imgs) == 2
    assert [f.size for f in sampled_imgs] == [(32, 24), (32, 24)]

    # Yield every 2nd decoded numpy frame from a video with backend='opencv'
    video = media.save_video(root / "clip.mp4", frames, fps=4.0)
    sampled_video = list(media.iter_frames(video, sample_rate=2, backend="opencv"))
    assert len(sampled_video) == 2 and sampled_video[0].shape == (24, 32, 3)
    print([f.size for f in sampled_imgs])
```
