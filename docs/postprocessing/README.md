# Postprocessing

`klygo.postprocessing` contains backend-neutral operations for `outputs.Detection` and `outputs.Detections`. Direct functions mutate the supplied object and return `None`. Factory functions return reusable `Operation` values. A `Pipeline` stages all work on detached copies and commits once, so an exception leaves the original result unchanged.

```python
from klygo import postprocessing as post

pipeline = post.compose(
    post.clip(),
    post.threshold(0.35),
    post.nms(iou=0.5),
    post.top(20),
)
pipeline(results)
```

## API reference

| API | Purpose |
| :--- | :--- |
| [`Operation`](api/Operation.md) | Wraps one named in-place operation. Calling it mutates the supplied result and returns `None`. |
| [`Pipeline`](api/Pipeline.md) | Stores ordered operations and applies them as one atomic transaction for `Detection` and `Detections`. |
| [`compose`](api/compose.md) | Builds a reusable atomic `Pipeline` from operation callables. |
| [`apply`](api/apply.md) | Applies operations immediately to the supplied result and returns `None`. |
| [`custom`](api/custom.md) | Wraps a caller-owned in-place callable as a named `Operation`. |
| [`select`](api/select.md) | Keeps boxes that match every supplied selector. It mutates a frame or every frame in a collection. |
| [`map_boxes`](api/map_boxes.md) | Runs a transformation on detached box drafts and commits all changes atomically. |
| [`relabel`](api/relabel.md) | Renames or merges labels and rebuilds the active class registry. |
| [`drop`](api/drop.md) | Removes boxes whose labels occur in `labels`. |
| [`filter_score`](api/filter_score.md) | Removes boxes with confidence below `minimum`. |
| [`clip_boxes`](api/clip_boxes.md) | Clamps coordinates to an image boundary and removes boxes wholly outside it. |
| [`remove_invalid_boxes`](api/remove_invalid_boxes.md) | Removes non-finite or undersized boxes according to `min_area`. |
| [`box_iou`](api/box_iou.md) | Returns the intersection-over-union value for two boxes without mutation. |
| [`suppress`](api/suppress.md) | Runs greedy score-ordered non-maximum suppression per frame. |
| [`top_boxes`](api/top_boxes.md) | Keeps at most `count` highest-confidence boxes per frame. |
| [`threshold`](api/threshold.md) | Creates an `Operation` that applies `filter_score`. |
| [`clip`](api/clip.md) | Creates an `Operation` that applies `clip_boxes`. |
| [`remove_invalid`](api/remove_invalid.md) | Creates an `Operation` that applies `remove_invalid_boxes`. |
| [`nms`](api/nms.md) | Creates an `Operation` that applies `suppress`. |
| [`top`](api/top.md) | Creates an `Operation` that applies `top_boxes`. |

## Mutation contract

- `Detection` operations commit a complete replacement only after validation succeeds.
- `Detections` operations apply to detached frame copies and commit the collection once.
- A composed `Pipeline` is atomic across all of its stages.
- Box identity (`uid`) and frame identity (`source_id`, `frame_index`) remain stable for retained items.
- `box_iou()` is the only direct numerical query and returns a `float`.

See the runnable scripts in [`examples/postprocessing`](../../examples/postprocessing/) and the behavioral tests in [`test/postprocessing`](../../test/postprocessing/).
