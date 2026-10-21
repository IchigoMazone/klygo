# `postprocessing.clip`

```python
postprocessing.clip(image_size: 'Optional[Tuple[int, int]]' = None) -> 'Operation'
```

Build a reusable box-clipping operation.

**Parameters:** image_size optionally overrides source dimensions.

**Returns and effects:** Returns a new result, an `Operation`, a `Pipeline`, or an IoU float as indicated by the signature. Detection transformations preserve the original result and map lazily over `Detections` streams.

**Errors:** Invalid result types raise `TypeError`. Invalid selectors or parameters raise their documented validation errors where applicable.

```python
from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(image, [Box(0, "cat", 0.9, [-5, 0, 50, 50]), Box(0, "cat", 0.8, [0, 0, 48, 48]), Box(1, "dog", 0.4, [60, 10, 90, 60])])
print(post.clip()(frame)[0].box)
```

Run [the standalone example](../../../examples/postprocessing/clip.py) or explore the [module index](../README.md).
