# `outputs.detect`

```python
outputs.detect.Box(
    id,
    label,
    score,
    box,
    parent_image=None,
    pad=0,
    uid=None,
    metadata=None,
    *,
    class_id=None,
)
outputs.detect.Crops(crops, source_image=None, pad=0, names=None)
outputs.detect.Detection(
    source_image,
    objects,
    speed=None,
    image_frame_index=0,
    config=None,
    names=None,
    **kwargs,
)
outputs.detect.Detections(
    frames,
    source_type="video",
    fps=30.0,
    output_path=None,
    stream=False,
    total_frames=None,
    names=None,
)
```

Represent single bounding boxes, object crop collections, single-frame detections, and multi-frame or streaming detection sequences.

## Contract

`klygo.outputs.detect` defines the four-tier result hierarchy used across Klygo detection workflows:

1. **`Box`**: Represents one detected object in `[xmin, ymin, xmax, ymax]` pixel coordinates. `Box.id` is the frame-local index, `Box.class_id` indexes the class registry (stored in `box.metadata["class_id"]`), and `Box.uid` is a stable identifier preserved across functional copies (`with_label`, `with_score`, `with_pad`, `with_box`, `with_metadata`, `translate`, `scale`, `clip`). When `parent_image` is set, `.image` and `.crop(pad=...)` lazily crop the sub-image bounded to image dimensions.
2. **`Crops`**: Sequence container of `Box` objects supporting indexing, slicing, iteration, concatenation (`+`), filtering (`filter`, `select`, `drop`, `relabel`), sorting (`sort`), grouping (`group_by`), mapping (`map`), and classification dataset export (`export`, `save`). `Crops.names` returns the sparse `{class_id: label}` dictionary referenced by its boxes.
3. **`Detection`**: Binds a list of `Box` objects to `source_image`, deduplicating any colliding `Box.id` values and linking every box's `parent_image` to `source_image`. Provides geometry and postprocessing helpers (`select`, `drop`, `relabel`, `filter_score`, `clip`, `remove_invalid`, `nms`, `top`, `pipe`, `restore`), visualization (`plot`, `draw`, `show`, `save`, `to_array`, `to_numpy`, `to_bgr`), and serialization (`export`, `to_dict`). `Detection.names` returns the sparse class mapping present in that frame.
4. **`Detections`**: Manages a sequence of `Detection` frames in memory or lazily as a stream (`stream=True` or non-list iterable). Supports positive indexing and slicing on streams, frame-level transformations (`map`, `map_boxes`, `select`, `relabel`, `drop`, `filter_score`, `clip`, `remove_invalid`, `nms`, `top`, `pipe`, `restore`), random-access frame editing (`get_frame`, `replace_frame`, `update_frame`, `remove_frame`), context-manager resource cleanup (`with results:` / `close()`), and batch export (`save`, `export`, `crop`, `to_dict`). `Detections.names` returns the complete sorted `{class_id: label}` registry across the collection.

For backward compatibility, `detect` also exports aliases: `CropResult`, `DetectedObject`, and `CroppedObject` (`Box`); `Boxes` and `CropResults` (`Crops`); `DetectionResult` (`Detection`); and `DetectionResults` and `PreviewResult` (`Detections`).

## Parameters

- `Box(id, label, score, box, parent_image=None, pad=0, uid=None, metadata=None, *, class_id=None)`:
  - `id`: `int` — frame-local box identifier.
  - `label`: `str | None` — class name (stripped; blank strings normalize to `None`).
  - `score`: `float` — confidence score.
  - `box`: `list[float]` — four pixel coordinates `[xmin, ymin, xmax, ymax]`.
  - `parent_image`: `PIL.Image.Image | None`, default `None` — source image used for lazy cropping.
  - `pad`: `int`, default `0` — pixel padding added around crops.
  - `uid`: `str | None`, default `None` — unique identifier; generated via `uuid.uuid4().hex` when omitted.
  - `metadata`: `dict[str, Any] | None`, default `None` — arbitrary key-value attributes.
  - `class_id`: `int | None`, default `None` — vocabulary class ID; defaults to `metadata["class_id"]` or `id`.
- `Crops(crops, source_image=None, pad=0, names=None)`:
  - `crops`: `list[Box]` — collection of `Box` objects.
  - `source_image`: `PIL.Image.Image | None`, default `None` — reference source image.
  - `pad`: `int`, default `0` — default padding for crops.
  - `names`: `Mapping[int, Any] | Sequence[Any] | None`, default `None` — class registry.
- `Detection(source_image, objects, speed=None, image_frame_index=0, config=None, names=None, **kwargs)`:
  - `source_image`: `PIL.Image.Image | Any` — frame image or `media.LazyImage`.
  - `objects`: `list[Box]` — detected boxes in this frame.
  - `speed`: `dict[str, float] | None`, default `None` — stage timing in ms (`preprocess`, `inference`, `postprocess`, `total`).
  - `image_frame_index`: `int`, default `0` — zero-based frame index.
  - `config`: `dict[str, Any] | None`, default `None` — frame configuration dictionary merged with `**kwargs`.
  - `names`: `Mapping[int, Any] | Sequence[Any] | None`, default `None` — class registry.
- `Detections(frames, source_type="video", fps=30.0, output_path=None, stream=False, total_frames=None, names=None)`:
  - `frames`: `list[Detection] | Iterable[Detection]` — frame collection or generator.
  - `source_type`: `str`, default `"video"` — source kind (`"image"`, `"video"`, `"directory"`, etc.).
  - `fps`: `float`, default `30.0` — frame rate for video export.
  - `output_path`: `str | None`, default `None` — default output path.
  - `stream`: `bool`, default `False` — keep `frames` as a lazy iterator until materialized.
  - `total_frames`: `int | None`, default `None` — known frame count for streams.
  - `names`: `Mapping[int, Any] | Sequence[Any] | None`, default `None` — base class vocabulary.

## Returns

Instances of `Box`, `Crops`, `Detection`, or `Detections` representing the corresponding level of the detection hierarchy.

## Errors and edge cases

- `Box.with_box()` raises `ValueError` if `coordinates` does not contain exactly 4 values.
- `Box.scale()` raises `ValueError` if `factor < 0`.
- `Box.clip()` raises `ValueError` if `image_size` is omitted and `parent_image` has no `.size`.
- `Detection.restore()` raises `ValueError` if `target` is not `"transformed"` or `"original"`, or if the source image has no file path.
- `Detections` in streaming mode raises `IndexError` for negative integer indices or out-of-range stream indices, and `ValueError` for non-positive slice steps or negative slice bounds.
- Concatenating `Detections` (`+`) or accessing `Detections.names` raises `ValueError` when the same `class_id` maps to conflicting labels across frames.

## Example

See [`detect.py`](../../../examples/outputs/detect.py) for an executable example.

## Tests

See [`test_detect.py`](../../../test/outputs/test_detect.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.outputs.detect."""

from PIL import Image
from klygo import outputs

image = Image.new("RGB", (80, 60), "white")
box = outputs.detect.Box(0, "item", 0.92, [10, 10, 40, 35], class_id=3)
frame = outputs.detect.Detection(image, [box], names={3: "item", 5: "unused"})
results = outputs.detect.Detections([frame], source_type="image", names={3: "item", 5: "unused"})

print("Frames:", len(results))
print("First crop size:", results[0][0].image.size)
print("Frame sparse names:", frame.names)
print("Batch complete names:", results.names)

shifted = box.translate(dx=5, dy=5).clip((80, 60))
assert shifted.box == [15.0, 15.0, 45.0, 40.0]
assert frame.names == {3: "item"}
assert results.names == {3: "item", 5: "unused"}
```
