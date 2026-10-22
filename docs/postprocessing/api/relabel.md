# `postprocessing.relabel`

```python
postprocessing.relabel(result: 'Result', label: 'Union[str, Mapping[int, str], Callable[[Box], str]]', *, ids: 'Optional[Iterable[int]]' = None, class_ids: 'Optional[Iterable[int]]' = None, uids: 'Optional[Iterable[str]]' = None, labels: 'Optional[Iterable[str]]' = None, where: 'Optional[Predicate]' = None) -> 'Result'
```

Change labels for selected boxes while retaining the rest.

**Parameters:** label is a string, ID mapping, or callable; selectors restrict affected boxes.

**Returns and effects:** Returns a new result, an `Operation`, a `Pipeline`, or an IoU float as indicated by the signature. Detection transformations preserve the original result and map lazily over `Detections` streams.

**Errors:** Invalid result types raise `TypeError`. Invalid selectors or parameters raise their documented validation errors where applicable.

```python
from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection

image = Image.new("RGB", (100, 80), "white")
frame = Detection(image, [Box(0, "cat", 0.9, [-5, 0, 50, 50]), Box(0, "cat", 0.8, [0, 0, 48, 48]), Box(1, "dog", 0.4, [60, 10, 90, 60])])
print(post.relabel(frame, "feline", labels=["cat"]).labels)
print(post.relabel(frame, "vehicle", class_ids=[1]).labels)
```

Run [the standalone example](../../../examples/postprocessing/relabel.py) or explore the [module index](../README.md).
