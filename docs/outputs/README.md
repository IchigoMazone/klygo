# `klygo.outputs`

`klygo.outputs` provides a three-level detection hierarchy: `Box`, `Detection`, and `Detections`. The same `Detections` type supports ordinary materialized results, compact packed arrays with lazy frame facades, and bounded-memory one-pass streams.

```python
from klygo import models, postprocessing as post

model = models.load("yolov8n.pt")
results = model.predict("images/", stream=False)
results.restore()  # Safe no-op when predict already restored the coordinates.
results.astype(32)
results.transform(post.clip(), post.threshold(0.4), post.nms(0.5))
results[0].save("artifacts/predictions/selected.jpg", mode="uid", overwrite=True)
results.save_video("artifacts/predictions/result.mp4", fps=30.0, mode="index", overwrite=True)
results.export("artifacts/predictions", format="yolo")
```

## Public APIs

- [`Box`](api/Box.md) stores one validated annotation with a stable frame-local UID.
- [`Detection`](api/Detection.md) owns the boxes for one `(source_id, frame_index)` identity and can restore that frame's coordinates.
- [`Detections`](api/Detections.md) owns ordered frames, history, class aggregation, indexed coordinate restoration, export, and reprediction.
- `Detections(..., packed=True)` and `model.predict(..., packed=True)` store numeric annotations in shared arrays while preserving `Detection` and `Box` access.
- [`build_detections`](api/build_detections.md) converts normalized backend predictions into frames.
- [`detect`](api/detect.md) contains the implementation of the three levels.

Postprocessing mutates `Detection` or `Detections` in place. Visual save methods decode one frame at a time; video export therefore remains bounded-memory for streams. Annotation export writes labels and source references without copying images.

`Detections` owns one storage profile for every frame. Use `results.cuda(0)` for
CuPy arrays, `results.cpu()` for NumPy arrays, and `results.astype(16|32|64)` for
floating coordinates and scores. Device and precision conversion remain lazy for
streams. Install optional CUDA storage with `pip install "klygo[gpu-outputs]"`.
