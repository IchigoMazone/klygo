# `postprocessing.Pipeline`

```python
postprocessing.Pipeline(operations: 'Iterable[Callable[[Any], Any]]') -> 'None'
```

Apply an ordered series of result transformations.

**Parameters:** operations is an iterable of callables or nested pipelines.

**Returns and effects:** Returns a new result, an `Operation`, a `Pipeline`, or an IoU float as indicated by the signature. Detection transformations preserve the original result and map lazily over `Detections` streams.

**Errors:** Invalid result types raise `TypeError`. Invalid selectors or parameters raise their documented validation errors where applicable.

```python
from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(image, [Box(0, "cat", 0.9, [-5, 0, 50, 50]), Box(0, "cat", 0.8, [0, 0, 48, 48]), Box(1, "dog", 0.4, [60, 10, 90, 60])])
pipeline = post.Pipeline([post.threshold(0.5), post.top(1)])
print(pipeline(frame).labels)
```

Run [the standalone example](../../../examples/postprocessing/Pipeline.py) or explore the [module index](../README.md).
