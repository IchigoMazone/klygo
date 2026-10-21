# `outputs.detect`

```python
outputs.detect.Box(id, label, score, box, parent_image=None, pad=0,
                   uid=None, metadata=None)
outputs.detect.Crops(crops, source_image=None, pad=0)
outputs.detect.Detection(source_image, objects, speed=None,
                         image_frame_index=0, config=None, **kwargs)
outputs.detect.Detections(frames, source_type="video", fps=30.0,
                          output_path=None, stream=False, total_frames=None)
```

`Box` describes one object with `[xmin, ymin, xmax, ymax]` pixel coordinates.
When `parent_image` is set, its `image` property crops lazily. `Crops` holds a
collection of boxes. `Detection` binds boxes to one image, setting their parent
image. `Detections` holds frames in memory or incrementally when `stream=True`.
These constructors return the corresponding objects. `Box.with_box` raises
`ValueError` unless given exactly four coordinates; negative scale also raises
`ValueError`. The module retains `Boxes`, `DetectionResult`, and other aliases
for existing callers.

```python
from PIL import Image
from klygo import outputs

image = Image.new("RGB", (40, 30))
box = outputs.detect.Box(0, "cat", 0.9, [5, 5, 20, 20])
frame = outputs.detect.Detection(image, [box])
results = outputs.detect.Detections([frame], source_type="image")
print(results[0][0].image.size)
```

See [the runnable example](../../../examples/outputs/detect.py) and
[`klygo.visual`](../../visual/README.md) for annotating images.
