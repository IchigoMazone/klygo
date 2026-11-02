# Postprocessing

`klygo.postprocessing` contains backend-neutral operation builders for `outputs.Detection` and `outputs.Detections`. Apply operations through `result.transform(...)`. A `Pipeline` stages all work on detached copies and commits once, so an exception leaves the original result unchanged. Structural changes such as `drop()` and `relabel()` remain methods of the result containers.

```python
from klygo import postprocessing as post

pipeline = post.compose(
    post.clip(),
    post.threshold(0.35),
    post.nms(iou=0.5),
    post.top(20),
)
results.transform(pipeline)
```

## API reference

| API | Purpose |
| :--- | :--- |
| [`Operation`](api/Operation.md) | Wraps one named in-place operation. Calling it mutates the supplied result and returns `None`. |
| [`Pipeline`](api/Pipeline.md) | Stores ordered operations and applies them as one atomic transaction for `Detection` and `Detections`. |
| [`compose`](api/compose.md) | Builds a reusable atomic `Pipeline` from operation callables. |
| [`custom`](api/custom.md) | Wraps a caller-owned in-place callable as a named `Operation`. |
| [`select`](api/select.md) | Creates an operation that keeps boxes matching every supplied selector. |
| [`map_boxes`](api/map_boxes.md) | Creates an operation that transforms detached box drafts atomically. |
| [`box_iou`](api/box_iou.md) | Returns the intersection-over-union value for two boxes without mutation. |
| [`threshold`](api/threshold.md) | Creates a score-threshold operation. |
| [`clip`](api/clip.md) | Creates a coordinate-clipping operation. |
| [`remove_invalid`](api/remove_invalid.md) | Creates an invalid-box removal operation. |
| [`nms`](api/nms.md) | Creates a non-maximum-suppression operation. |
| [`top`](api/top.md) | Creates a highest-confidence selection operation. |

## Mutation contract

- Apply operations with `Detection.transform()` or `Detections.transform()`.
- Use container methods for CRUD, `drop()`, and `relabel()`.
- A composed `Pipeline` is atomic across all of its stages.
- Box identity (`uid`) and frame identity (`source_id`, `frame_index`) remain stable for retained items.
- `box_iou()` is the only direct numerical query and returns a `float`.

See the runnable scripts in [`examples/postprocessing`](../../examples/postprocessing/) and the behavioral tests in [`test/postprocessing`](../../test/postprocessing/).
