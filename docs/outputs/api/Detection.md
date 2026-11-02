# `outputs.Detection`

```python
Detection(boxes=None, *, source_id=None, frame_index=0, source_path=None, image_size=None, source=None, metadata=None, names=None)
```

Represents one source frame. It provides positional access, stable UID lookup, atomic box CRUD, frame-level `filter`, `map`, `drop`, `relabel`, and `restore`, three display caption modes, and annotation serialization.

`names` contains only active classes and maps each active `class_id` to its label. Adding, replacing, relabeling, or deleting boxes reconciles this mapping before the mutation commits.

Use `preview_box(...)` to display the current frame with a proposed box drawn in red before adding it. The method returns the annotated image and does not mutate the frame.

```python
frame.preview_box([20, 30, 160, 190], label="car", score=0.9)
frame.add_box([20, 30, 160, 190], label="car", score=0.9)
```

```python
from klygo import outputs
frame = outputs.Detection([], source_id="sample", frame_index=0, image_size=(100, 80))
frame.add_box([5, 5, 30, 25], label="car", score=0.8)
frame.show(mode="uid")
```

`show()` supports `default`, `index`, and `uid`. It does not change annotations.

Pass a list of box positions to create an independent subset while retaining
the frame identity, source handle, metadata, UIDs, precision, and restore trace.
Annotation arrays and metadata are copied; lazy source pixels remain shared.

```python
sub_frame = frame[[5, 7, 9]]
sub_frame.show(mode="uid")
```

Duplicate or invalid positions raise rather than creating ambiguous duplicate
UIDs. Integer indexing still returns one `Box`, and slicing still returns a
tuple of boxes.

`save()` uses the same `default`, `index`, and `uid` captions as `show()` and
writes one annotated image:

```python
frame.save("artifacts/predictions/frame.jpg", mode="index", overwrite=True)
```

`restore(trace=None)` maps every box in-place from processed coordinates to the
original image coordinates. By default it reads the preprocessing trace carried
by the source image. Passing `trace` supports results produced by an external
model. A frame without a trace, or one already restored, is a safe no-op.

```python
frame.restore()
```

`model.predict()` already performs this restoration automatically, so calling
`restore()` on its result is harmless and does not transform coordinates twice.

`device` and `dtype` report the frame's array storage but are read-only at this
level. The owning [`Detections`](Detections.md) collection controls storage for
every frame together through `cuda()`, `cpu()`, and `astype()`. This invariant
prevents a collection from mixing NumPy and CuPy frames or numeric precision.

`filter()`, `drop()`, `relabel()`, and `clear()` mutate the frame and return the
same instance for method chaining.
