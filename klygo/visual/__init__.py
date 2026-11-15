"""Non-destructive visualization, adaptive display, and dataset analytics charts.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/visual/Document_Visual_Overview.ipynb

Public API (4 functions):
    1. draw_bboxes(...)        Render bounding boxes and labels onto an image copy.
    2. plot_dataset_stats(...) Plot class distribution and box-geometry statistics.
    3. show_image(...)         Display an image in notebook, GUI, or terminal environments.
    4. compare_images(...)     Render side-by-side image comparisons.
"""

from .draw_bboxes import draw_bboxes, draw_box
from .plot_dataset_stats import plot_dataset_stats
from .show_image import show_image
from .compare_images import compare_images

__all__ = ["draw_bboxes", "plot_dataset_stats", "show_image", "compare_images"]
