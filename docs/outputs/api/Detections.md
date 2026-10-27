# `outputs.Detections`

```python
Detections(frames=(), *, metadata=None, names=None, stream=False)
```

Owns either an ordered materialized frame collection or a bounded-memory one-pass frame stream. Materialized results provide collision-safe CRUD, stable frame lookup, coordinate restoration, net change properties, mutation history, class-wide `drop` and `relabel`, explicit class-ID `compact`, backend-neutral `repredict`, JSON serialization, and annotation-only YOLO export.

`model.predict(..., stream=True)` returns a lazy `Detections`. Iterate it directly to process every frame without retaining earlier frames. Random access, `len()`, copying, reprediction, and export require `stream=False` because one-pass results intentionally do not cache yielded frames.

```python
source = media.load("video.mp4", stream=True)
stream = model.predict(source)
stream.transform(postprocessing.compose(postprocessing.threshold(0.4), postprocessing.nms(0.5)))
for frame in stream:
    frame.show()

results = model.predict("images/", stream=False)
results.drop("unknown")
results.export("artifacts/predictions", format="yolo")
```

`restore(index=None, trace=None)` restores box coordinates in-place. Omit
`index` to process every frame, pass one integer for one list position, or pass
an iterable such as `[1, 3]`. Repeated positions are processed once. Index
selection is validated before any frame changes. A one-pass stream supports
only `index=None` and restores each frame lazily as it is yielded.

```python
results.restore(index=0)
results.restore(index=[1, 3])
results.restore()
```

Each frame normally uses its own source trace. An explicit `trace` applies the
same preprocessing history to every selected frame. Frames already restored by
`model.predict()` are skipped safely.

Visual exports draw annotations without changing them. `save_images()` writes a
numbered image sequence. `save_video()` passes a generator directly to the media
encoder, so a streaming collection remains bounded-memory:

```python
results.save_images("artifacts/predictions/frames", prefix="frame", mode="uid")
results.save_video("artifacts/predictions/result.mp4", fps=30.0, mode="index")
```

`is_stream` identifies one-pass mode, `stream_exhausted` reports completion, and `close()` releases the underlying media iterator early. Iteration is single-consumer.

The collection owns the shared class registry. Frame CRUD and direct mutations on an owned `Detection` automatically reconcile every active label so the same label has one `class_id` across all frames. Removed labels disappear from `names`.

The collection maintains an internal label-to-frame index. Collection-wide `drop()` and `relabel()` therefore visit only frames containing a requested label and automatically batch registry synchronization. Callers use the normal methods directly; batching remains an implementation detail.

Class IDs remain stable and deleted IDs are not reused during normal editing. Call `compact()` explicitly to rewrite active IDs to `0..n-1`. Compaction updates the shared registries lazily instead of visiting every box, so its cost depends on the number of classes and frames rather than the total annotation count. Box UIDs and frame identities remain unchanged. YOLO export performs its own temporary compact mapping and never mutates the collection.

Use `copy()` to create an independent working result before in-place filtering. Frames, boxes, registries, and metadata are copied deeply, while lazy source-image handles are shared so image pixels are not duplicated in memory.

```python
raw_results = model.predict(images)
cleaned_results = raw_results.copy()
cleaned_results.transform(
    postprocessing.threshold(0.4),
    postprocessing.nms(0.5),
)
```


```python
from klygo import models, outputs
model = models.load("yolov8n.pt")
results = model.predict("images/", stream=False)
results.drop(["unknown"])
results.repredict(model, index=[0, 2], threshold=0.5)
results.compact()
results.export("artifacts/predictions", format="yolo")
```

`merge()` raises on any duplicate `(source_id, frame_index)` identity. Batch operations validate before committing. `filter()`, `drop()`, `relabel()`, `compact()`, and `clear()` return the same collection for in-place chaining:

```python
results.drop("bg").relabel({"car": "vehicle"}).compact()
```

The collection owns one array storage profile shared by every frame. `cuda(device=0)`
moves annotation arrays to CuPy, `cpu()` moves them to NumPy, and `astype(16|32|64)`
changes floating coordinate and score precision. Class IDs remain `int32`; stable box
UIDs remain `int64`. All methods mutate in place and return the same collection for
chaining. Install CUDA output support with `pip install "klygo[gpu-outputs]"`.

Materialized collections validate and prepare every frame before committing a
conversion. Streams queue conversions and apply them lazily as frames are yielded.
Storage conversions are recorded in `history` but do not mark frames as annotation
edits in `modified_frames`. Individual `Detection` frames expose read-only `device`
and `dtype` properties and cannot change storage independently:

```python
compact_results = results.copy()
compact_results.cuda(0).astype(16)
print(compact_results.device, compact_results.dtype)  # cuda:0 16

compact_results.astype(32).cpu()
print(compact_results.device, compact_results.dtype)  # cpu 32
```

GPU postprocessing runs directly on CuPy arrays. Visual display, image/video save,
JSON serialization, and YOLO export stage only the current frame on CPU and do not
change the collection's storage profile. Float16 storage uses float32 intermediates
for sorting, NMS, and area calculations to support older CUDA devices and reduce
geometry error.
