# `postprocessing.Pipeline`

```python
postprocessing.Pipeline(operations)
```

Execute an ordered sequence of detection transformations while preserving lazy `Detections` streams.

## Contract

Validates and normalizes `operations` into an immutable tuple stored on `self.operations`:
- Any nested `Pipeline` instances inside `operations` are flattened in-order into their constituent callables.
- Every element must be callable (such as an [`Operation`](Operation.md) or standard Python function).
- Calling `pipeline(result)` applies each operation left-to-right. When `result` is a `Detections` collection or lazy stream, `pipeline(result)` delegates to `result.map(self._apply_detection)` so video and directory streams remain lazy until iterated.
- Supports `len(pipeline)`, iteration (`for op in pipeline:`), and human-readable `repr(pipeline)` (`Pipeline(threshold, top)`).

## Parameters

- `operations`: `Iterable[Callable[[Any], Any]]` — iterable of callable result transformations or nested `Pipeline` objects.

## Returns

`Pipeline` — reusable callable sequence of postprocessing stages.

## Errors and edge cases

Raises `TypeError("Every postprocessing operation must be callable")` during initialization if any item in `operations` is not callable. An empty `operations` iterable produces a valid no-op `Pipeline` of length `0` that returns `result` unchanged (or mapped as a no-op over `Detections`).

## Example

See [`Pipeline.py`](../../../examples/postprocessing/Pipeline.py) for a standalone runnable script.

## Complete executable example

```python
"""Build, nest, inspect, and execute a postprocessing Pipeline."""

from PIL import Image
from klygo import postprocessing as post
from klygo.outputs.detect import Box, Detection, Detections

image = Image.new("RGB", (100, 80), "white")
frame = Detection(
    image,
    [
        Box(0, "cat", 0.9, [-5, 0, 50, 50]),
        Box(0, "cat", 0.8, [0, 0, 48, 48]),
        Box(1, "dog", 0.4, [60, 10, 90, 60]),
    ],
)

stage_a = post.Pipeline([post.clip(), post.threshold(0.5)])
pipeline = post.Pipeline([stage_a, post.nms(0.5), post.top(1)])
assert len(pipeline) == 4
assert repr(pipeline) == "Pipeline(clip, threshold, nms, top)"

single_out = pipeline(frame)
assert single_out.labels == ["cat"] and single_out[0].box == [0.0, 0.0, 50.0, 50.0]

batch_out = pipeline(Detections([frame]))
assert len(batch_out) == 1 and batch_out[0].labels == ["cat"]
print(single_out.labels)
```
