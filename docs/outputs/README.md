# `klygo.outputs` Documentation

`klygo.outputs` converts normalized backend predictions into structured, multi-level object-detection result hierarchies (`Box`, `Crops`, `Detection`, and `Detections`) with lazy image cropping, coordinate restoration, and in-memory or streaming frame evaluation. Import the public facade with `from klygo import outputs`.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/outputs/Document_Outputs_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/outputs/`, 1-to-1 with `klygo/outputs/` submodules):
- [`Document_Outputs_Overview.ipynb`](../../tutorials/outputs/Document_Outputs_Overview.ipynb)
- [`Document_Outputs_Detect.ipynb`](../../tutorials/outputs/Document_Outputs_Detect.ipynb) (`klygo/outputs/detect.py`)

---

## 1. Core Concepts & Data Structures

<details open>
<summary><b>Hierarchical detection structures and identity model</b> <i>(Click to expand or collapse)</i></summary>

`klygo.outputs` models object-detection predictions across four composable levels exposed through `outputs.detect`:

| Class / Function | Granularity | Key Responsibilities | Identity & Vocabulary Behavior |
| :--- | :--- | :--- | :--- |
| **`Box`** | Single object | Stores `id`, `class_id`, `uid`, `label`, `score`, and `[xmin, ymin, xmax, ymax]` pixel coordinates. Computes geometry (`width`, `height`, `area`, `center`, `aspect_ratio`) and crops from `parent_image` lazily via `.image` or `.crop(pad=...)`. Supports immutable functional updates (`with_label`, `with_score`, `with_box`, `translate`, `scale`, `clip`). | `Box.id` is the frame-local index (`0..N-1`); `Box.class_id` indexes the model vocabulary; `Box.uid` is a UUID hex string preserved across copies and transformations. |
| **`Crops`** | Object collection | Sequence-like container of `Box` instances extracted from one or more frames. Supports slicing, filtering (`filter`, `select`, `drop`, `relabel`), sorting (`sort`), grouping (`group_by`), and exporting cropped sub-images into classification folders (`export`, `save`). | `.names` returns the sparse `{class_id: label}` mapping actually present in the crop collection. |
| **`Detection`** | Single image / frame | Binds a list of `Box` objects to a `source_image`, timing dictionary (`speed`), `image_frame_index`, and open configuration metadata (`config`). Provides postprocessing shortcuts (`select`, `drop`, `relabel`, `filter_score`, `nms`, `top`, `clip`, `restore`), rendering (`plot`, `draw`, `show`, `save`, `to_array`, `to_bgr`), and dataset export (`export`, `to_dict`). | Automatically deduplicates colliding `Box.id` values within the frame. `Detection.names` returns the sparse mapping of classes present in that frame. |
| **`Detections`** | Batch or video stream | Holds a list or lazy iterator of `Detection` frames (`stream=True` or generator input). Supports frame-level indexing, positive stream slicing, functional transformations across frames (`map`, `select`, `drop`, `relabel`, `filter_score`, `nms`, `restore`), frame editing (`get_frame`, `replace_frame`, `update_frame`, `remove_frame`), and batch video/dataset export (`save`, `export`, `crop`). | `Detections.names` aggregates the complete vocabulary across the base registry and all materialized frames, raising `ValueError` on conflicting class ID assignments. |
| **`build_detections()`** | Prediction builder | Converts backend-neutral prediction dictionaries (`boxes`, `scores`, `labels` / `text_labels`, optional `class_ids`) and source images into a `list[Detection]`. | Assigns sequential frame-local `Box.id` values (`0..N-1`) and resolves `Box.class_id` against `names`, appending newly discovered labels in first-seen order. |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (16 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Build `Detection` frames from raw backend predictions** | [`build_detections()`](api/build_detections.md) | `frames = outputs.build_detections(images, preds, names={0: "cat"})` |
| **Construct a single bounding box manually** | [`detect.Box`](api/detect.md) | `box = outputs.detect.Box(0, "cat", 0.95, [10, 20, 100, 150], class_id=0)` |
| **Create an immutable copy of a box with updated coordinates or label** | [`detect.Box`](api/detect.md) | `box2 = box.with_label("tiger").translate(dx=5, dy=-5).scale(1.1)` |
| **Lazily crop a detected object from its parent image** | [`detect.Box`](api/detect.md) | `crop_img = frame[0].crop(pad=8)` or `crop_img = frame[0].image` |
| **Wrap boxes and an image into a single-frame result** | [`detect.Detection`](api/detect.md) | `frame = outputs.detect.Detection(image, [box], names={0: "cat"})` |
| **Inspect sparse frame classes vs. full batch vocabulary** | [`detect.Detection`](api/detect.md) / [`detect.Detections`](api/detect.md) | `frame.names` (sparse per frame) vs. `results.names` (complete registry) |
| **Filter or relabel boxes on a frame or batch** | [`detect.Detection`](api/detect.md) | `clean = results.filter_score(0.5).select(labels=["cat"]).nms(iou=0.45)` |
| **Restore boxes to original pre-letterbox image coordinates** | [`detect.Detection`](api/detect.md) | `restored = frame.restore(target="original")` |
| **Render annotated bounding boxes onto a PIL image** | [`detect.Detection`](api/detect.md) | `annotated_pil = frame.plot(line_width=2, labels=True, scores=True)` |
| **Convert annotated or raw frame to RGB / BGR NumPy array** | [`detect.Detection`](api/detect.md) | `bgr = frame.to_bgr(annotated=True)` or `rgb = frame.to_array()` |
| **Export frame or batch detections to YOLO `.txt` or JSON** | [`detect.Detection`](api/detect.md) / [`detect.Detections`](api/detect.md) | `results.export("output_dataset", format="yolo")` |
| **Extract all object crops into a classification folder tree** | [`detect.Crops`](api/detect.md) | `results.crop(output_path="crops_dir", pad=4)` |
| **Consume video detections lazily without exhausting RAM** | [`detect.Detections`](api/detect.md) | `stream_res = outputs.detect.Detections(frame_gen, stream=True, fps=30.0)` |
| **Slice or transform a streaming detection sequence lazily** | [`detect.Detections`](api/detect.md) | `sub_stream = stream_res[10:100:2].filter_score(0.6)` |
| **Replace or re-run inference on a specific video frame index** | [`detect.Detections`](api/detect.md) | `updated = results.update_frame(42, lambda f: f.filter_score(0.8))` |
| **Save annotated frames to a video file or image directory** | [`detect.Detections`](api/detect.md) | `results.save("annotated_output.mp4", fps=30.0)` |

</details>

---

## 3. Public APIs Reference (1 Function, 1 Submodule across `detect.py`)

<details open>
<summary><b>Detection result builders and hierarchy — <code>detect.py</code> (2 exports)</b> <i>(Click to expand)</i></summary>

- [`build_detections(images, predictions, *, metadata=None, names=None)`](api/build_detections.md): Build one `Detection` per image from normalized backend prediction mappings ([example](../../examples/outputs/build_detections.py)).
- [`detect`](api/detect.md): Submodule providing `Box`, `Crops`, `Detection`, `Detections`, `build_detections`, and backward-compatible aliases (`Boxes`, `CropResult`, `CropResults`, `DetectedObject`, `CroppedObject`, `DetectionResult`, `DetectionResults`, `PreviewResult`) ([example](../../examples/outputs/detect.py)).

</details>

---

## 4. End-to-End Detection & Cropping Workflow

```python
from PIL import Image
from klygo import outputs

image = Image.new("RGB", (120, 80), "white")
frames = outputs.build_detections(
    [image],
    [
        {
            "boxes": [[10, 10, 50, 45], [60, 20, 110, 70]],
            "scores": [0.94, 0.42],
            "labels": ["person", "bag"],
            "class_ids": [0, 24],
        }
    ],
    names={0: "person", 24: "bag", 26: "handbag"},
)

results = outputs.detect.Detections(frames, source_type="image", names={0: "person", 24: "bag", 26: "handbag"})
filtered = results.filter_score(0.5)

assert filtered[0].names == {0: "person"}
assert filtered.names == {0: "person", 24: "bag", 26: "handbag"}
assert filtered[0][0].crop(pad=2).size == (44, 39)
```
