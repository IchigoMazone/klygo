# `klygo.media` Documentation

`klygo.media` provides 15 public APIs (12 functions and 3 classes) for zero-RAM lazy image and video loading, bounded-memory frame streaming, format conversions across PIL, NumPy, and PyTorch, metadata inspection, and video/image serialization.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/media/Document_Media_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/media/`, 1-to-1 with `klygo/media/` submodules):
- [`Document_Media_Overview.ipynb`](../../tutorials/media/Document_Media_Overview.ipynb)
- [`Document_Media_Operations.ipynb`](../../tutorials/media/Document_Media_Operations.ipynb) (`klygo/media/operations.py`)

---

## 1. Core Concepts & Lazy / Composable Pipeline Overview

<details open>
<summary><b>Zero-RAM proxies, indexed frame containers, bounded streams, and format converters</b> <i>(Click to expand or collapse)</i></summary>

`klygo.media` separates metadata discovery and pipeline attachment from pixel decoding so large image directories and high-resolution videos can be inspected, sliced, cropped, and filtered before allocating frame buffers in RAM:

1. **[`LazyImage`](api/LazyImage.md) (Zero-RAM Pixel Proxy)**:
   Subclasses `PIL.Image.Image` and wraps a file path or video frame index without decoding pixels on construction. Reading `.size`, `.width`, `.height`, or `.shape` inspects headers or cached dimensions only. Calling `.crop()` or `.lazy_crop()` records a bounding box in zero RAM, and `.process()` attaches `klygo.preprocessing` operations lazily until `.load()`, `.to_pil()`, `.to_array()`, or `.to_tensor()` materializes the image.
2. **[`MediaFrames`](api/MediaFrames.md) (Unified Frame Container)**:
   Returned by [`load()`](api/load.md) in both modes. With `stream=False`, it is an indexed list-like collection of `LazyImage` proxies. With `stream=True`, it becomes a bounded-memory sequential iterator with `sample_rate` and `max_frames` controls. Both modes share `.transform()`, `.where()`, `.where_quality()`, `.where_motion()`, `.deduplicate()`, `.sample()`, and `.limit()`.
3. **[`VideoReader`](api/VideoReader.md) (Smart Sequential & Random-Access Seeker)**:
   Manages a single reusable `cv2.VideoCapture` handle with position tracking so consecutive reads avoid redundant seeks while out-of-order frame indices seek on demand.
4. **Format Conversions & Media I/O**:
   [`to_array()`](api/to_array.md), [`to_pil()`](api/to_pil.md), and [`to_tensor()`](api/to_tensor.md) normalize any `LazyImage`, `PIL.Image.Image`, `numpy.ndarray`, or `torch.Tensor` into an independent target representation. [`save()`](api/save.md), [`save_images()`](api/save_images.md), [`save_video()`](api/save_video.md), [`convert()`](api/convert.md), [`copy()`](api/copy.md), and [`probe()`](api/probe.md) handle persistence and header inspection using `klygo.files` path semantics.

| Container / Type | Decodes on Init? | Random Indexing | Best For |
| :--- | :--- | :--- | :--- |
| [`LazyImage`](api/LazyImage.md) | No | N/A (single frame) | Deferred decoding, zero-RAM crops, lazy preprocessing |
| [`MediaFrames`](api/MediaFrames.md) | No | Materialized mode only | Indexed inspection or bounded-memory streaming |
| [`VideoReader`](api/VideoReader.md) | Header only | Yes (`read_frame(i)`) | Low-level indexed BGR frame reads from a video file |

</details>

---

## 2. Task-Oriented Cheatsheet

<details open>
<summary><b>Task-oriented quick reference (18 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Load an image, video, or folder lazily** | [`load()`](api/load.md) | `frames = media.load("video.mp4")` |
| **Stream a long video with bounded RAM** | [`load()`](api/load.md) / [`MediaFrames`](api/MediaFrames.md) | `frames = media.load("video.mp4", stream=True, sample_rate=5)` |
| **Inspect image or video metadata** | [`probe()`](api/probe.md) | `meta = media.probe("clip.mp4", verbose=True)` |
| **Check dimensions without decoding pixels** | [`LazyImage`](api/LazyImage.md) | `img = media.LazyImage("photo.jpg"); print(img.size)` |
| **Crop a region without loading into RAM** | [`LazyImage.lazy_crop()`](api/LazyImage.md) | `patch = frames[0].lazy_crop((10, 10, 110, 90))` |
| **Filter stream frames by visual quality** | [`MediaFrames.where_quality()`](api/MediaFrames.md) | `frames.where_quality(min_sharpness=20.0, max_blur=0.6)` |
| **Filter stream frames by scene motion** | [`MediaFrames.where_motion()`](api/MediaFrames.md) | `frames.where_motion(threshold=0.03)` |
| **Drop near-duplicate consecutive frames** | [`MediaFrames.deduplicate()`](api/MediaFrames.md) | `frames.deduplicate(threshold=0.95)` |
| **Attach preprocessing lazily to frames** | [`MediaFrames.transform()`](api/MediaFrames.md) | `frames.transform(preprocessing.resize((640, 640)))` |
| **Iterate decoded frames directly** | [`iter_frames()`](api/iter_frames.md) | `for frame in media.iter_frames("clip.mp4", sample_rate=2): ...` |
| **Read specific video frame by index** | [`VideoReader`](api/VideoReader.md) | `reader = media.VideoReader("clip.mp4"); bgr = reader.read_frame(10)` |
| **Convert any image to a NumPy array** | [`to_array()`](api/to_array.md) | `arr = media.to_array(frames[0])` |
| **Convert any image to a CHW PyTorch tensor** | [`to_tensor()`](api/to_tensor.md) | `tensor = media.to_tensor(frames[0], normalize=True)` |
| **Convert any image to a PIL Image** | [`to_pil()`](api/to_pil.md) | `pil_img = media.to_pil(arr)` |
| **Save a single image to disk** | [`save()`](api/save.md) | `media.save("out/result.png", frames[0], overwrite=True)` |
| **Save a sequence of images as numbered files** | [`save_images()`](api/save_images.md) | `media.save_images("out/frames", frames, prefix="img", extension=".jpg")` |
| **Encode frames into an MP4 or AVI video** | [`save_video()`](api/save_video.md) | `media.save_video("out/clip.mp4", frames, fps=30.0, overwrite=True)` |
| **Convert image format or transcode video** | [`convert()`](api/convert.md) | `media.convert("input.png", "output.webp", overwrite=True)` |
| **Copy a media file or directory safely** | [`copy()`](api/copy.md) | `media.copy("source.mp4", "backup/source.mp4", overwrite=True)` |

</details>

---

## 3. Public APIs Reference (12 Functions, 3 Classes in `operations.py`)

<details open>
<summary><b>Loading, streaming, and inspection — <code>operations.py</code> (4 functions)</b> <i>(Click to expand)</i></summary>

- [`load(source, recursive=False, backend='pil', verbose=False, *, stream=False, sample_rate=1, max_frames=None)`](api/load.md) ([example](../../examples/media/load.py)): Load an image, video, or directory into [`MediaFrames`](api/MediaFrames.md) in indexed or streaming mode.
- [`compose(*steps)`](api/compose.md) ([example](../../examples/media/compose.py)): Chain filtering, sampling, and transform steps into a reusable media pipeline.
- [`iter_frames(source, sample_rate=1, recursive=False, backend='pil', verbose=False)`](api/iter_frames.md) ([example](../../examples/media/iter_frames.py)): Yield sampled frames from a video or image directory as a generator.
- [`probe(path, *, verbose=False)`](api/probe.md) ([example](../../examples/media/probe.py)): Read image or video format, dimensions, codec, and timing metadata into a dictionary (with optional Rich table display).

</details>

<details open>
<summary><b>Saving, transcoding, and copying — <code>operations.py</code> (5 functions)</b> <i>(Click to expand)</i></summary>

- [`save(path, image, overwrite=False, verbose=False)`](api/save.md) ([example](../../examples/media/save.py)): Save a `PIL.Image.Image`, `numpy.ndarray`, or `LazyImage` to disk, creating parent directories automatically.
- [`save_video(output_path, frames, fps=30.0, fourcc='mp4v', overwrite=False, verbose=False)`](api/save_video.md) ([example](../../examples/media/save_video.py)): Encode an iterable of frames into a video file via OpenCV `VideoWriter`.
- [`save_images(output_dir, images, prefix='frame', extension='.jpg', overwrite=False, verbose=False)`](api/save_images.md) ([example](../../examples/media/save_images.py)): Write an iterable of images to numbered files (`{prefix}_{idx:06d}{extension}`).
- [`convert(source, target, codec=None, crf=23, fps=None, gpu=None, preset=None, overwrite=False, verbose=False)`](api/convert.md) ([example](../../examples/media/convert.py)): Convert between image formats or transcode video files via FFmpeg (with OpenCV fallback).
- [`copy(source, target, overwrite=False)`](api/copy.md) ([example](../../examples/media/copy.py)): Copy a media file or directory tree to a target path via `klygo.files.copy`.

</details>

<details open>
<summary><b>In-memory format conversions — <code>operations.py</code> (3 functions)</b> <i>(Click to expand)</i></summary>

- [`to_array(image)`](api/to_array.md) ([example](../../examples/media/to_array.py)): Convert a `LazyImage`, `PIL.Image.Image`, `numpy.ndarray`, or `torch.Tensor` into an independent `numpy.ndarray`.
- [`to_tensor(image, normalize=True)`](api/to_tensor.md) ([example](../../examples/media/to_tensor.py)): Convert supported image data into a CHW `torch.Tensor` (requires optional `torch`).
- [`to_pil(image)`](api/to_pil.md) ([example](../../examples/media/to_pil.py)): Convert supported image data into an independent `PIL.Image.Image`.

</details>

<details open>
<summary><b>Lazy containers and frame proxies — <code>operations.py</code> (3 classes)</b> <i>(Click to expand)</i></summary>

- [`LazyImage(path, backend='pil', crop_box=None, frame_index=None, reader=None)`](api/LazyImage.md) ([example](../../examples/media/LazyImage.py)): Zero-RAM image and video-frame proxy supporting lazy crops, PIL method forwarding, and preprocessing pipelines.
- [`MediaFrames(source, *, stream=False, sample_rate=1, max_frames=None, recursive=False, backend='pil', verbose=False)`](api/MediaFrames.md) ([example](../../examples/media/MediaFrames.py)): Unified indexed and streaming frame container with quality, motion, deduplication, and preprocessing stages.
- [`VideoReader(path)`](api/VideoReader.md) ([example](../../examples/media/VideoReader.py)): Reusable OpenCV video capture wrapper for sequential and random-access frame decoding.

</details>
