# `klygo.postprocessing` Documentation

`klygo.postprocessing` provides 20 public APIs (18 functions and 2 classes across `core.py` and `detect.py`) for composable, non-mutating transformations on normalized Klygo object-detection results (`Detection`, `Detections`, and `Crops`).

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/postprocessing/Document_Postprocessing_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/postprocessing/`, 1-to-1 with `klygo/postprocessing/` submodules):
- [`Document_Postprocessing_Overview.ipynb`](../../tutorials/postprocessing/Document_Postprocessing_Overview.ipynb)
- [`Document_Postprocessing_Core.ipynb`](../../tutorials/postprocessing/Document_Postprocessing_Core.ipynb) (`klygo/postprocessing/core.py`)
- [`Document_Postprocessing_Detect.ipynb`](../../tutorials/postprocessing/Document_Postprocessing_Detect.ipynb) (`klygo/postprocessing/detect.py`)

---

## 1. Core Concepts & Lazy / Composable Pipeline Overview

<details open>
<summary><b>Immutable result transforms, lazy stream mapping, and functional vs. pipeline APIs</b> <i>(Click to expand or collapse)</i></summary>

`klygo.postprocessing` separates immediate one-shot result transformations from reusable, declarative pipeline stages while guaranteeing immutability and lazy video stream preservation:

1. **Core Composition Primitives (`core.py`)**:
   - [`Operation(name, function)`](api/Operation.md): A frozen dataclass wrapping a callable transformation with a descriptive name (`Operation(threshold)`).
   - [`Pipeline(operations)`](api/Pipeline.md): Flattens nested `Pipeline` instances and applies a sequence of callables left-to-right. When called on a `Detections` collection or stream, `Pipeline` delegates to `Detections.map(...)` so lazy video streams remain lazy until iterated.
   - [`compose(*operations)`](api/compose.md) & [`apply(result, *operations)`](api/apply.md): `compose()` builds a reusable `Pipeline` from variadic callables or a single iterable; `apply()` executes one or more operations immediately on a result.
   - [`custom(function, name=None)`](api/custom.md): Wraps any Python function `Detection -> Detection` into a named `Operation`.
2. **Direct Functional Box Operations vs. Pipeline Operation Builders (`detect.py`)**:
   - **Direct functions** take `result` (`Detection`, `Detections`, or `Crops`) as their first argument and return a new transformed result immediately without mutating the input: [`select()`](api/select.md), [`drop()`](api/drop.md), [`relabel()`](api/relabel.md), [`map_boxes()`](api/map_boxes.md), [`filter_score()`](api/filter_score.md), [`clip_boxes()`](api/clip_boxes.md), [`remove_invalid_boxes()`](api/remove_invalid_boxes.md), [`suppress()`](api/suppress.md), and [`top_boxes()`](api/top_boxes.md).
   - **Pipeline builders** take configuration parameters and return a reusable [`Operation`](api/Operation.md) ready for [`compose()`](api/compose.md), [`Pipeline`](api/Pipeline.md), or [`apply()`](api/apply.md): [`threshold()`](api/threshold.md), [`clip()`](api/clip.md), [`remove_invalid()`](api/remove_invalid.md), [`nms()`](api/nms.md), and [`top()`](api/top.md).
   - **Geometry helper**: [`box_iou(first, second)`](api/box_iou.md) computes exact intersection-over-union for two `Box` objects or `[xmin, ymin, xmax, ymax]` sequences.

| Transformation Goal | Direct Function (`result` first) | Pipeline Factory (returns `Operation`) |
| :--- | :--- | :--- |
| **Score filtering** | [`filter_score(result, minimum)`](api/filter_score.md) | [`threshold(minimum)`](api/threshold.md) |
| **Coordinate clipping** | [`clip_boxes(result, image_size=None)`](api/clip_boxes.md) | [`clip(image_size=None)`](api/clip.md) |
| **Invalid / tiny box removal** | [`remove_invalid_boxes(result, min_area=0)`](api/remove_invalid_boxes.md) | [`remove_invalid(min_area=0)`](api/remove_invalid.md) |
| **Non-maximum suppression** | [`suppress(result, iou=0.5, class_agnostic=False)`](api/suppress.md) | [`nms(iou=0.5, class_agnostic=False)`](api/nms.md) |
| **Top-$K$ highest scores** | [`top_boxes(result, count)`](api/top_boxes.md) | [`top(count)`](api/top.md) |

</details>

---

## 2. Task-Oriented Cheatsheet

<details open>
<summary><b>Task-oriented quick reference (18 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Build a reusable multi-step postprocessing pipeline** | [`compose()`](api/compose.md) | `pipe = post.compose(post.remove_invalid(), post.threshold(0.5), post.nms(0.45))` |
| **Apply multiple steps to a result in one call** | [`apply()`](api/apply.md) | `clean = post.apply(frame, post.clip(), post.threshold(0.4), post.top(10))` |
| **Wrap a custom function as a named pipeline step** | [`custom()`](api/custom.md) | `op = post.custom(lambda r: post.select(r, labels=["car"]), name="only_cars")` |
| **Keep boxes matching labels, IDs, or a predicate** | [`select()`](api/select.md) | `cats = post.select(frame, labels=["cat"], where=lambda b: b.score >= 0.6)` |
| **Remove unwanted classes or specific boxes** | [`drop()`](api/drop.md) | `filtered = post.drop(frame, labels=["background", "noise"])` |
| **Rename labels by string, dict mapping, or callable** | [`relabel()`](api/relabel.md) | `renamed = post.relabel(frame, "vehicle", labels=["car", "truck"])` |
| **Transform or selectively drop each `Box`** | [`map_boxes()`](api/map_boxes.md) | `mapped = post.map_boxes(frame, lambda b: b.with_label("target") if b.score > 0.5 else None)` |
| **Filter boxes by minimum confidence directly** | [`filter_score()`](api/filter_score.md) | `confident = post.filter_score(frame, 0.5)` |
| **Create a confidence threshold pipeline stage** | [`threshold()`](api/threshold.md) | `step = post.threshold(0.5); clean = step(frame)` |
| **Clamp box coordinates to image boundaries** | [`clip_boxes()`](api/clip_boxes.md) / [`clip()`](api/clip.md) | `clipped = post.clip_boxes(frame, image_size=(640, 480))` |
| **Discard NaN, inverted, or tiny bounding boxes** | [`remove_invalid_boxes()`](api/remove_invalid_boxes.md) | `valid = post.remove_invalid_boxes(frame, min_area=16.0)` |
| **Create an invalid-box filter stage for a pipeline** | [`remove_invalid()`](api/remove_invalid.md) | `step = post.remove_invalid(min_area=16.0)` |
| **Run score-ordered Non-Maximum Suppression (NMS)** | [`suppress()`](api/suppress.md) / [`nms()`](api/nms.md) | `deduped = post.suppress(frame, iou=0.5, class_agnostic=False)` |
| **Keep only the highest-scoring $K$ detections** | [`top_boxes()`](api/top_boxes.md) / [`top()`](api/top.md) | `best = post.top_boxes(frame, 5)` |
| **Compute IoU between two boxes or coordinate lists** | [`box_iou()`](api/box_iou.md) | `overlap = post.box_iou(frame[0], [0, 0, 50, 50])` |
| **Inspect or iterate over stages in a pipeline** | [`Pipeline`](api/Pipeline.md) | `pipe = post.Pipeline([post.clip(), post.top(3)]); print(len(pipe), list(pipe))` |
| **Construct a named operation explicitly** | [`Operation`](api/Operation.md) | `op = post.Operation("keep_dogs", lambda r: post.select(r, labels=["dog"]))` |

</details>

---

## 3. Public APIs Reference (18 Functions, 2 Classes across 2 Submodules)

<details open>
<summary><b>Pipeline composition and execution — <code>core.py</code> (3 functions, 2 classes)</b> <i>(Click to expand)</i></summary>

- [`Operation(name, function)`](api/Operation.md) ([example](../../examples/postprocessing/Operation.py)): Immutable named callable wrapping a single result transformation.
- [`Pipeline(operations)`](api/Pipeline.md) ([example](../../examples/postprocessing/Pipeline.py)): Ordered sequence of transformations that flattens nested pipelines and maps lazily over `Detections`.
- [`compose(*operations)`](api/compose.md) ([example](../../examples/postprocessing/compose.py)): Construct a `Pipeline` from variadic callables or a single iterable of operations.
- [`apply(result, *operations)`](api/apply.md) ([example](../../examples/postprocessing/apply.py)): Immediately apply zero or more operations in sequence to a `Detection`, `Detections`, or `Crops` object.
- [`custom(function, name=None)`](api/custom.md) ([example](../../examples/postprocessing/custom.py)): Wrap a custom callable into a named `Operation` instance.

</details>

<details open>
<summary><b>Direct detection and box transformations — <code>detect.py</code> (10 functions)</b> <i>(Click to expand)</i></summary>

- [`select(result, *, ids=None, class_ids=None, uids=None, labels=None, where=None)`](api/select.md) ([example](../../examples/postprocessing/select.py)): Keep only boxes that satisfy every supplied selector criterion.
- [`drop(result, *, ids=None, class_ids=None, uids=None, labels=None, where=None)`](api/drop.md) ([example](../../examples/postprocessing/drop.py)): Remove boxes that match all supplied selector criteria.
- [`relabel(result, label, *, ids=None, class_ids=None, uids=None, labels=None, where=None)`](api/relabel.md) ([example](../../examples/postprocessing/relabel.py)): Replace the label of matching boxes using a constant string, ID-to-label mapping, or callable.
- [`map_boxes(result, function)`](api/map_boxes.md) ([example](../../examples/postprocessing/map_boxes.py)): Apply `function(box)` to each `Box`, keeping returned `Box` instances and dropping `None` results.
- [`filter_score(result, minimum)`](api/filter_score.md) ([example](../../examples/postprocessing/filter_score.py)): Retain boxes with `box.score >= minimum`.
- [`clip_boxes(result, image_size=None)`](api/clip_boxes.md) ([example](../../examples/postprocessing/clip_boxes.py)): Clamp box coordinates to `(width, height)` or the frame's own `image_size`.
- [`remove_invalid_boxes(result, min_area=0)`](api/remove_invalid_boxes.md) ([example](../../examples/postprocessing/remove_invalid_boxes.py)): Drop boxes with non-finite coordinates, non-positive width/height, or `area < min_area`.
- [`box_iou(first, second)`](api/box_iou.md) ([example](../../examples/postprocessing/box_iou.py)): Compute intersection-over-union in $[0.0, 1.0]$ between two `Box` objects or 4-element coordinate sequences.
- [`suppress(result, iou=0.5, class_agnostic=False)`](api/suppress.md) ([example](../../examples/postprocessing/suppress.py)): Perform score-ordered non-maximum suppression per class or class-agnostically.
- [`top_boxes(result, count)`](api/top_boxes.md) ([example](../../examples/postprocessing/top_boxes.py)): Keep at most `count` boxes per frame sorted by descending confidence score.

</details>

<details open>
<summary><b>Reusable pipeline operation builders — <code>detect.py</code> (5 functions)</b> <i>(Click to expand)</i></summary>

- [`threshold(minimum)`](api/threshold.md) ([example](../../examples/postprocessing/threshold.py)): Return an `Operation("threshold", ...)` that calls `filter_score(result, minimum)`.
- [`clip(image_size=None)`](api/clip.md) ([example](../../examples/postprocessing/clip.py)): Return an `Operation("clip", ...)` that calls `clip_boxes(result, image_size)`.
- [`remove_invalid(min_area=0)`](api/remove_invalid.md) ([example](../../examples/postprocessing/remove_invalid.py)): Return an `Operation("remove_invalid", ...)` that calls `remove_invalid_boxes(result, min_area)`.
- [`nms(iou=0.5, class_agnostic=False)`](api/nms.md) ([example](../../examples/postprocessing/nms.py)): Return an `Operation("nms", ...)` that calls `suppress(result, iou, class_agnostic)`.
- [`top(count)`](api/top.md) ([example](../../examples/postprocessing/top.py)): Return an `Operation("top", ...)` that calls `top_boxes(result, count)`.

</details>
