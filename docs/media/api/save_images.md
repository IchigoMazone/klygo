# `media.save_images`

```python
media.save_images(output_dir, images, prefix='frame', extension='.jpg', overwrite=False, verbose=False)
```

Save an iterable of images or lazy frames to sequentially numbered files inside a directory.

## Contract

Resolves and creates `output_dir` via `klygo.files.path()` and `klygo.files.mkdir()`. Normalizes `extension` so it always begins with a leading dot (e.g., `'png'` becomes `'.png'`). Iterates through `images` with 1-based indexing and writes each item to `{prefix}_{idx:06d}{extension}` (for example, `frame_000001.jpg`, `frame_000002.jpg`) by delegating to [`media.save()`](save.md).

Because each item is saved sequentially as it is yielded, `images` can be [`MediaFrames`](MediaFrames.md) in either mode, a list of `PIL.Image.Image` or `numpy.ndarray` objects, or any generator.

## Parameters

- `output_dir`: `str | os.PathLike[str]` — directory where numbered image files will be written.
- `images`: `Iterable[PIL.Image.Image | numpy.ndarray | LazyImage]` — sequence or stream of images to save.
- `prefix`: `str`, default `'frame'` — filename stem prefix before the `_000001` counter.
- `extension`: `str`, default `'.jpg'` — file extension (with or without a leading `.`).
- `overwrite`: `bool`, default `False` — allow replacing existing numbered files in `output_dir`.
- `verbose`: `bool`, default `False` — display a batch progress bar.

## Returns

`list[pathlib.Path]` — ordered list of paths written to disk. Returns an empty list if `images` is empty.

## Errors and edge cases

Raises `TypeError` for invalid argument types or unsupported image elements, `FileExistsError` if any target file already exists while `overwrite=False`, and `ValueError` if an image cannot be encoded to `extension`.

## Example

See [`save_images.py`](../../../examples/media/save_images.py) for a standalone runnable script.

## Complete executable example

```python
"""Write a batch of frames to numbered image files in an output directory."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    frames = [Image.new("RGB", (32, 24), color) for color in ("red", "green", "blue")]

    paths = media.save_images(
        root / "frames",
        frames,
        prefix="sample",
        extension="png",
    )
    names = [files.name(p) for p in paths]
    assert names == ["sample_000001.png", "sample_000002.png", "sample_000003.png"]

    # Reload the saved directory lazily
    reloaded = media.load(root / "frames")
    assert len(reloaded) == 3 and reloaded[0].size == (32, 24)
    print(names)
```
