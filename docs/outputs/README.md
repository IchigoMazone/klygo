# `klygo.outputs`

`klygo.outputs` provides a materialized three-level detection hierarchy: `Box`, `Detection`, and `Detections`.

```python
from klygo import models, postprocessing as post

model = models.load("yolov8n.pt")
results = model.predict("images/", stream=False)
post.apply(results, post.clip(), post.threshold(0.4), post.nms(0.5))
results[0].show(mode="uid")
results.export("artifacts/predictions", format="yolo")
```

## Public APIs

- [`Box`](api/Box.md) stores one validated annotation with a stable frame-local UID.
- [`Detection`](api/Detection.md) owns the boxes for one `(source_id, frame_index)` identity.
- [`Detections`](api/Detections.md) owns ordered frames, history, class aggregation, export, and reprediction.
- [`build_detections`](api/build_detections.md) converts normalized backend predictions into frames.
- [`detect`](api/detect.md) contains the implementation of the three levels.

Postprocessing mutates `Detection` or `Detections` in place. Pixel decoding is limited to `Detection.show()` and model inference. Export writes annotations and source references without copying images.
