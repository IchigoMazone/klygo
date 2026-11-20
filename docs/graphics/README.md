# `klygo.graphics` Documentation

`klygo.graphics` provides non-destructive bounding-box annotation, adaptive multi-backend image display across notebooks and desktop environments, and YOLO dataset class-distribution plotting. Import the public facade with `from klygo import graphics`.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/graphics/Document_Graphics_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/graphics/`, 1-to-1 with `klygo/graphics/` submodules):
- [`Document_Graphics_Overview.ipynb`](../../tutorials/graphics/Document_Graphics_Overview.ipynb)
- [`Document_Graphics_Draw_Bboxes.ipynb`](../../tutorials/graphics/Document_Graphics_Draw_Bboxes.ipynb) (`klygo/graphics/draw.py`)
- [`Document_Graphics_Show_Image.ipynb`](../../tutorials/graphics/Document_Graphics_Show_Image.ipynb) (`klygo/graphics/show.py`)
- [`Document_Graphics_Plot_Dataset_Stats.ipynb`](../../tutorials/graphics/Document_Graphics_Plot_Dataset_Stats.ipynb) (`klygo/graphics/dataset_stats.py`)

---

## 1. Core Concepts & Architecture

<details open>
<summary><b>Non-destructive rendering, adaptive backends, and dataset inspection</b> <i>(Click to expand or collapse)</i></summary>

`klygo.graphics` separates visual presentation into four focused submodules that interoperate with `klygo.media`, `klygo.outputs`, and `klygo.datasets`:

| Submodule | Primary API | Input / Output Representation | Key Architectural Behavior |
| :--- | :--- | :--- | :--- |
| **`draw.py`** | [`draw()`](api/draw.md) | Accepts `PIL.Image.Image`, RGB `numpy.ndarray`, or `media.LazyImage` (`.to_pil()`) and returns an annotated copy in the same representation (`PIL.Image.Image` or `numpy.ndarray`). | **Two-pass rendering**: draws all bounding-box outlines in Pass 1 and renders filled label/score badges in Pass 2 with collision avoidance so badges do not obscure each other. Assigns deterministic colors per unique class label from a 20-color Ultralytics palette unless `box_color` is overridden. |
| **`show.py`** | [`show()`](api/show.md) | Accepts `PIL.Image.Image` or RGB `numpy.ndarray` and returns `None`. | **Environment-aware display**: `backend="auto"` detects Jupyter, Google Colab, and Kaggle shells (`ZMQInteractiveShell`, `Shell`, `google.colab`) to render inline via `IPython.display.display`, falling back to `PIL.Image.show()` on desktop. Also supports `backend="matplotlib"` (`figsize` in inches) and OpenCV (`cv2.imshow`). |
| **`dataset_stats.py`** | [`dataset_stats()`](api/dataset_stats.md) | Accepts a YOLO dataset directory or archive path (`str` or `pathlib.Path`) and returns `None`. | **Archive-ready class histogram**: queries `datasets.info()` for class metadata, extracts archives into a temporary directory when needed, scans label files under `images/` and `labels/`, and plots per-class object counts with Matplotlib. |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (12 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Draw XYXY bounding boxes on a PIL image or RGB NumPy array** | [`draw()`](api/draw.md) | `out = graphics.draw(img, [[10, 10, 80, 60]], labels=["cat"], scores=[0.94])` |
| **Draw boxes without modifying the original input image** | [`draw()`](api/draw.md) | `annotated = graphics.draw(image, bboxes)` (always copies input) |
| **Override box border color and caption text color** | [`draw()`](api/draw.md) | `graphics.draw(img, bboxes, labels=["car"], box_color=(0, 200, 100), text_color=(0, 0, 0))` |
| **Customize border width and label font size** | [`draw()`](api/draw.md) | `graphics.draw(img, bboxes, line_width=3, font_size=16)` |
| **Hide class labels or confidence scores in box badges** | [`draw()`](api/draw.md) | `graphics.draw(img, bboxes, labels=["dog"], scores=[0.88], show_scores=False)` |
| **Display an image automatically in Jupyter/Colab or desktop** | [`show()`](api/show.md) | `graphics.show(img, title="Preview", backend="auto")` |
| **Resize an image proportionally for display only** | [`show()`](api/show.md) | `graphics.show(img, width=480)` |
| **Render an image inside a Matplotlib figure** | [`show()`](api/show.md) | `graphics.show(img, title="Prediction", backend="matplotlib", figsize=(8, 8))` |
| **Open an interactive OpenCV window until a key is pressed** | [`show()`](api/show.md) | `graphics.show(img, title="OpenCV Window", backend="opencv")` |
| **Plot class frequency bar chart for a YOLO dataset folder** | [`dataset_stats()`](api/dataset_stats.md) | `graphics.dataset_stats("datasets/coco128", figsize=(10, 6))` |
| **Plot class frequency directly from a `.zip` or `.tar` dataset archive** | [`dataset_stats()`](api/dataset_stats.md) | `graphics.dataset_stats("datasets/coco128.zip")` |
| **Combine `outputs.Detection` with `graphics` helpers** | [`draw()`](api/draw.md) / [`show()`](api/show.md) | `graphics.show(graphics.draw(det.source_image, det.boxes, det.labels, det.scores))` |

</details>

---

## 3. Public APIs Reference (4 Functions across 4 Submodules)

<details open>
<summary><b>Bounding-box annotation — <code>draw.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`draw(image, bboxes, labels=None, scores=None, thickness=None, line_width=None, font_size=None, box_color=None, text_color=None, show_labels=True, show_scores=True, **kwargs)`](api/draw.md): Draw pixel-coordinate `[xmin, ymin, xmax, ymax]` bounding boxes and collision-aware caption badges on a copy of an image ([example](../../examples/graphics/draw.py)).

</details>

<details open>
<summary><b>Image display — <code>show.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`show(image, title='Image', width=None, backend='auto', figsize=(10, 10))`](api/show.md): Display a PIL image or RGB NumPy array inline in notebooks, in a desktop viewer, or via Matplotlib/OpenCV ([example](../../examples/graphics/show.py)).

</details>

<details open>
<summary><b>Dataset statistics — <code>dataset_stats.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`dataset_stats(source, figsize=(10, 6))`](api/dataset_stats.md): Plot per-class object counts from a YOLO dataset directory or archive as a Matplotlib bar chart ([example](../../examples/graphics/dataset_stats.py)).

</details>

## Side-by-side annotation comparison

Use [compare](api/compare.md) with two Detection frames; each supplies
its own image and annotations. Use mode="show" (default) to display the comparison
or mode="save" with output_path to save it without opening a viewer.
Frames may have different sources and dimensions, or share an image for model/GT comparison.

[Example](../../examples/graphics/compare.py) ·
[Guide](../../tutorials/graphics/Document_Graphics_Compare_Images.ipynb)
