# `postprocessing.nms`

```python
postprocessing.nms(iou: 'float' = 0.5, class_agnostic: 'bool' = False) -> 'Operation'
```

Build a reusable nonmaximum-suppression operation.

**Parameters:** iou is overlap threshold; class_agnostic optionally ignores class IDs.

**Returns and effects:** Returns a new result, an `Operation`, a `Pipeline`, or an IoU float as indicated by the signature. Detection transformations preserve the original result and map lazily over `Detections` streams.

**Errors:** Invalid result types raise `TypeError`. Invalid selectors or parameters raise their documented validation errors where applicable.

```python
from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(image, [Box(0, "cat", 0.9, [-5, 0, 50, 50]), Box(0, "cat", 0.8, [0, 0, 48, 48]), Box(1, "dog", 0.4, [60, 10, 90, 60])])
print(post.nms(iou=0.5)(frame).labels)
```

Run [the standalone example](../../../examples/postprocessing/nms.py) or explore the [module index](../README.md).
