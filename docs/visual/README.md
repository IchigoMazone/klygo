# `klygo.visual` Documentation

`klygo.visual` provides non-destructive bounding-box annotation, adaptive multi-backend image display across notebooks and desktop environments, and YOLO dataset class-distribution plotting. Import the public facade with `from klygo import visual`.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/visual/Document_Visual_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/visual/`, 1-to-1 with `klygo/visual/` submodules):
- [`Document_Visual_Overview.ipynb`](../../tutorials/visual/Document_Visual_Overview.ipynb)
- [`Document_Visual_Draw_Bboxes.ipynb`](../../tutorials/visual/Document_Visual_Draw_Bboxes.ipynb) (`klygo/visual/draw_bboxes.py`)
- [`Document_Visual_Show_Image.ipynb`](../../tutorials/visual/Document_Visual_Show_Image.ipynb) (`klygo/visual/show_image.py`)
- [`Document_Visual_Plot_Dataset_Stats.ipynb`](../../tutorials/visual/Document_Visual_Plot_Dataset_Stats.ipynb) (`klygo/visual/plot_dataset_stats.py`)

---

## 1. Core Concepts & Architecture

<details open>
<summary><b>Non-destructive rendering, adaptive backends, and dataset inspection</b> <i>(Click to expand or collapse)</i></summary>

`klygo.visual` separates visual presentation into three focused submodules that interoperate with `klygo.media`, `klygo.outputs`, and `klygo.datasets`:

| Submodule | Primary API | Input / Output Representation | Key Architectural Behavior |
| :--- | :--- | :--- | :--- |
| **`draw_bboxes.py`** | [`draw_bboxes()`](api/draw_bboxes.md) | Accepts `PIL.Image.Image`, RGB `numpy.ndarray`, or `media.LazyImage` (`.to_pil()`) and returns an annotated copy in the same representation (`PIL.Image.Image` or `numpy.ndarray`). | **Two-pass rendering**: draws all bounding-box outlines in Pass 1 and renders filled label/score badges in Pass 2 with collision avoidance so badges do not obscure each other. Assigns deterministic colors per unique class label from a 20-color Ultralytics palette unless `box_color` is overridden. |
| **`show_image.py`** | [`show_image()`](api/show_image.md) | Accepts `PIL.Image.Image` or RGB `numpy.ndarray` and returns `None`. | **Environment-aware display**: `backend="auto"` detects Jupyter, Google Colab, and Kaggle shells (`ZMQInteractiveShell`, `Shell`, `google.colab`) to render inline via `IPython.display.display`, falling back to `PIL.Image.show()` on desktop. Also supports `backend="matplotlib"` (`figsize` in inches) and OpenCV (`cv2.imshow`). |
| **`plot_dataset_stats.py`** | [`plot_dataset_stats()`](api/plot_dataset_stats.md) | Accepts a YOLO dataset directory or archive path (`str` or `pathlib.Path`) and returns `None`. | **Archive-ready class histogram**: queries `datasets.info()` for class metadata, extracts archives into a temporary directory when needed, scans label files under `images/` and `labels/`, and plots per-class object counts with Matplotlib. |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (12 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Draw XYXY bounding boxes on a PIL image or RGB NumPy array** | [`draw_bboxes()`](api/draw_bboxes.md) | `out = visual.draw_bboxes(img, [[10, 10, 80, 60]], labels=["cat"], scores=[0.94])` |
| **Draw boxes without modifying the original input image** | [`draw_bboxes()`](api/draw_bboxes.md) | `annotated = visual.draw_bboxes(image, bboxes)` (always copies input) |
| **Override box border color and caption text color** | [`draw_bboxes()`](api/draw_bboxes.md) | `visual.draw_bboxes(img, bboxes, labels=["car"], box_color=(0, 200, 100), text_color=(0, 0, 0))` |
| **Customize border width and label font size** | [`draw_bboxes()`](api/draw_bboxes.md) | `visual.draw_bboxes(img, bboxes, line_width=3, font_size=16)` |
| **Hide class labels or confidence scores in box badges** | [`draw_bboxes()`](api/draw_bboxes.md) | `visual.draw_bboxes(img, bboxes, labels=["dog"], scores=[0.88], show_scores=False)` |
| **Display an image automatically in Jupyter/Colab or desktop** | [`show_image()`](api/show_image.md) | `visual.show_image(img, title="Preview", backend="auto")` |
| **Resize an image proportionally for display only** | [`show_image()`](api/show_image.md) | `visual.show_image(img, width=480)` |
| **Render an image inside a Matplotlib figure** | [`show_image()`](api/show_image.md) | `visual.show_image(img, title="Prediction", backend="matplotlib", figsize=(8, 8))` |
| **Open an interactive OpenCV window until a key is pressed** | [`show_image()`](api/show_image.md) | `visual.show_image(img, title="OpenCV Window", backend="opencv")` |
| **Plot class frequency bar chart for a YOLO dataset folder** | [`plot_dataset_stats()`](api/plot_dataset_stats.md) | `visual.plot_dataset_stats("datasets/coco128", figsize=(10, 6))` |
| **Plot class frequency directly from a `.zip` or `.tar` dataset archive** | [`plot_dataset_stats()`](api/plot_dataset_stats.md) | `visual.plot_dataset_stats("datasets/coco128.zip")` |
| **Combine `outputs.Detection` with `visual` helpers** | [`draw_bboxes()`](api/draw_bboxes.md) / [`show_image()`](api/show_image.md) | `visual.show_image(visual.draw_bboxes(det.source_image, det.boxes, det.labels, det.scores))` |

</details>

---

## 3. Public APIs Reference (3 Functions across 3 Submodules)

<details open>
<summary><b>Bounding-box annotation — <code>draw_bboxes.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`draw_bboxes(image, bboxes, labels=None, scores=None, thickness=None, line_width=None, font_size=None, box_color=None, text_color=None, show_labels=True, show_scores=True, **kwargs)`](api/draw_bboxes.md): Draw pixel-coordinate `[xmin, ymin, xmax, ymax]` bounding boxes and collision-aware caption badges on a copy of an image ([example](../../examples/visual/draw_bboxes.py)).

</details>

<details open>
<summary><b>Image display — <code>show_image.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`show_image(image, title='Image', width=None, backend='auto', figsize=(10, 10))`](api/show_image.md): Display a PIL image or RGB NumPy array inline in notebooks, in a desktop viewer, or via Matplotlib/OpenCV ([example](../../examples/visual/show_image.py)).

</details>

<details open>
<summary><b>Dataset statistics — <code>plot_dataset_stats.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`plot_dataset_stats(source, figsize=(10, 6))`](api/plot_dataset_stats.md): Plot per-class object counts from a YOLO dataset directory or archive as a Matplotlib bar chart ([example](../../examples/visual/plot_dataset_stats.py)).

</details>
