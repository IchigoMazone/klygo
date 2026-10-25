# `outputs.detect`

The module contains `Box`, `Detection`, `Detections`, and `build_detections`.

```python
from PIL import Image
from klygo import outputs

image = Image.new("RGB", (80, 60), "white")
box = outputs.Box([10, 10, 40, 35], label="item", score=0.92)
frame = outputs.Detection([box], source_id="example", image_size=image.size, source=image)
results = outputs.Detections([frame])
```

Frame position and source identity are separate. Box position and UID are also separate. UIDs are monotonically assigned within each frame and surviving boxes are never renumbered.

All CRUD and transformation methods mutate their receiver. Batch methods validate complete candidate states before committing. See the class pages for signatures and examples.
