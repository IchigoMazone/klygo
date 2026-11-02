# `media.compose`

```python
media.compose(*steps)
```

Create a composite filtering and transformation pipeline for media streams and frame collections.

## Contract

Accepts a sequence of filtering steps, preprocessing operations, or custom callables either as positional arguments (`*steps`) or a single `list`/`tuple`. Returns a callable `MediaCompose` pipeline that applies each step sequentially to a media container:
- Applies preprocessing [`preprocessing.Operation`](../../preprocessing/README.md) and [`preprocessing.Compose`](../../preprocessing/README.md) instances via `target.transform(step)`.
- Applies named filter steps (`compose.quality`, `compose.motion`, `compose.deduplicate`, `compose.sample`, `compose.limit`, `compose.where`) to [`MediaFrames`](MediaFrames.md) in both modes.
- If passed a file or folder path string, automatically loads it with [`media.load(target)`](load.md) before executing the pipeline.

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `*steps` | `callable` or `iterable[callable]` | required | Ordered filtering steps, transforms, or pipelines. |

## Returns

`MediaCompose`
A callable pipeline applicable to [`MediaFrames`](MediaFrames.md) or media file paths.

## Built-in Step Helpers

| Helper | Description |
| :--- | :--- |
| `media.where(predicate)` | Filter frames using a custom predicate function. |
| `media.quality(**kwargs)` | Filter frames using image quality conditions (`min_brightness`, `max_blur`, etc.). |
| `media.motion(threshold=0.02)` | Filter frames by inter-frame motion score difference. |
| `media.deduplicate(threshold=0.95)` | Filter consecutive duplicate frames by visual similarity. |
| `media.sample(sample_rate=1)` | Sample every nth frame from stream or collection. |
| `media.limit(max_frames=None)` | Limit total frames yielded or collected. |

## Example

```python
from PIL import Image
import tempfile
from klygo import files, media, preprocessing as p

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    for index in range(5):
        image = Image.new("RGB", (32, 24), color=(index * 30, index * 30, index * 30))
        media.save(root / f"frame_{index}.png", image)

    pipeline = media.compose(
        media.quality(min_brightness=0.1),
        media.sample(1),
        p.resize((16, 16)),
    )

    with media.load(root, stream=True) as stream:
        processed_stream = pipeline(stream)
        stream_sizes = [frame.size for frame in processed_stream]

    frames = media.load(root, stream=False)
    processed_frames = pipeline(frames)
    frame_sizes = [frame.size for frame in processed_frames]

    assert stream_sizes == frame_sizes
```

See the runnable script in [`examples/media/compose.py`](../../../examples/media/compose.py) and test coverage in [`test/media/test_media_unified.py`](../../../test/media/test_media_unified.py).
