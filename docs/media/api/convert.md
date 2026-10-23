# `media.convert`

```python
media.convert(source, target, codec=None, crf=23, fps=None, gpu=None, preset=None, overwrite=False, verbose=False)
```

Convert an image file to another image format or transcode a video file to a new container or codec.

## Contract

Inspects the lowercase file extensions of `source` and `target` via `klygo.files.extension()`:
- **Image-to-image conversion** (both extensions in `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp`): loads `source` lazily with [`media.load()`](load.md) and writes the first frame to `target` via [`media.save()`](save.md), respecting `overwrite`.
- **Video-to-video transcoding** (both extensions in `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`): prefers `ffmpeg` when available on `PATH` (preserving audio streams with `-c:a copy`, applying `-crf` or NVENC `-cq`, and falling back from auto-detected GPU NVENC to CPU `libx264` if hardware encoding fails). When `ffmpeg` is unavailable, falls back to frame-by-frame streaming and OpenCV [`save_video()`](save_video.md).

Cross-kind conversions (image to video or video to image) are rejected; use [`save_video()`](save_video.md) or [`save_images()`](save_images.md) instead.

## Parameters

- `source`: `str | os.PathLike[str]` — existing source image or video path.
- `target`: `str | os.PathLike[str]` — destination image or video path.
- `codec`: `str | None`, default `None` — optional video codec name (`'h264'`, `'h265'`, `'vp9'`, `'mjpeg'`, or custom FourCC/FFmpeg codec).
- `crf`: `int`, default `23` — constant rate factor / quality target for FFmpeg video encoding.
- `fps`: `float | None`, default `None` — output video frame rate override; defaults to source FPS.
- `gpu`: `bool | None`, default `None` — force GPU NVENC (`True`), CPU (`False`), or auto-detect (`None`).
- `preset`: `str | None`, default `None` — FFmpeg encoder preset (defaults to `'p4'` on GPU or `'fast'` on CPU).
- `overwrite`: `bool`, default `False` — allow replacing an existing `target` file.
- `verbose`: `bool`, default `False` — display conversion progress.

## Returns

`pathlib.Path` — resolved path to the converted output file.

## Errors and edge cases

Raises `TypeError` for invalid argument types, `FileNotFoundError` when `source` does not exist, `FileExistsError` when `target` exists and `overwrite=False`, `ValueError` when `source` and `target` are not both images or both videos, and `RuntimeError` if OpenCV fallback cannot decode any video frames.

## Example

See [`convert.py`](../../../examples/media/convert.py) for a standalone runnable script.

## Complete executable example

```python
"""Convert images across formats and transcode video files."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    source = media.save(root / "source.png", Image.new("RGB", (32, 24), "red"))

    # Convert PNG to JPEG and WebP
    jpg_target = media.convert(source, root / "converted.jpg")
    webp_target = media.convert(source, root / "converted.webp")
    assert media.probe(jpg_target)["format"] == "JPEG"
    assert media.probe(webp_target)["format"] == "WEBP"

    # Re-convert with overwrite=True
    media.convert(source, jpg_target, overwrite=True)
    print(media.probe(jpg_target)["format"], media.probe(webp_target)["size"])
```
