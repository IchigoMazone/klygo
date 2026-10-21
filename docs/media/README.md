# `klygo.media`

Load, stream, convert, and save images and videos. Import with `from klygo import media, files`.

All public APIs live in `operations.py`. Video encoding uses OpenCV; tensor conversion requires optional PyTorch.

| API | Example |
| --- | --- |
| [`load`](api/load.md) | [Run](../../examples/media/load.py) |
| [`save`](api/save.md) | [Run](../../examples/media/save.py) |
| [`convert`](api/convert.md) | [Run](../../examples/media/convert.py) |
| [`copy`](api/copy.md) | [Run](../../examples/media/copy.py) |
| [`save_video`](api/save_video.md) | [Run](../../examples/media/save_video.py) |
| [`save_images`](api/save_images.md) | [Run](../../examples/media/save_images.py) |
| [`iter_frames`](api/iter_frames.md) | [Run](../../examples/media/iter_frames.py) |
| [`probe`](api/probe.md) | [Run](../../examples/media/probe.py) |
| [`to_array`](api/to_array.md) | [Run](../../examples/media/to_array.py) |
| [`to_tensor`](api/to_tensor.md) | [Run](../../examples/media/to_tensor.py) |
| [`to_pil`](api/to_pil.md) | [Run](../../examples/media/to_pil.py) |
| [`MediaFrames`](api/MediaFrames.md) | [Run](../../examples/media/MediaFrames.py) |
| [`MediaStream`](api/MediaStream.md) | [Run](../../examples/media/MediaStream.py) |
| [`LazyImage`](api/LazyImage.md) | [Run](../../examples/media/LazyImage.py) |
| [`VideoReader`](api/VideoReader.md) | [Run](../../examples/media/VideoReader.py) |

Follow the [overview notebook](../../tutorials/media/Document_Media_Overview.ipynb) and [operations guide](../../tutorials/media/Document_Media_Operations.ipynb).
