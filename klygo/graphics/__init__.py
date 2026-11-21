"""Non-destructive visualization, adaptive display, composable grids, and report charts.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/graphics/Document_Graphics_Overview.ipynb

Public API (14 functions):
    Image rendering, display & export:
        1. draw(...)          Render bounding boxes and labels onto an image copy.
        2. show(...)          Display an image in notebook, GUI, or terminal environments.
        3. close(...)         Close one or more Matplotlib figures to free resources.
        4. compare(...)       Render side-by-side Detection comparisons.
        5. save(...)          Export a Matplotlib figure to PNG, PDF, or SVG.

    Composable grid layout:
        5. cell(...)          Configure a single cell for compose() and grid().
        6. compose(...)       Normalize heterogeneous cell sources into images.
        7. grid(...)          Arrange cells or images into a rows x cols grid.

    Charts:
        8. dataset_stats(...) Plot class distribution and object counts.
        9. partitions(...)    Plot image counts and percentages per split.
       10. curves(...)        Plot training or validation loss/metric curves.
       11. thresholds(...)    Plot P/R/F1 curves or TP/FP counts by threshold.
       12. mining(...)        Plot mining positive/background or confidence bars.
       13. bars(...)          Plot single-metric (± std) or multi-metric bars.
"""

from .draw_bboxes import draw
from .plot_dataset_stats import dataset_stats
from .show_image import show, close
from .compare_images import compare
from .layout import cell, compose, grid
from .charts import (
    partitions,
    curves,
    thresholds,
    mining,
    bars,
)
from ._render import save

__all__ = [
    "draw",
    "show",
    "close",
    "compare",
    "save",
    "cell",
    "compose",
    "grid",
    "dataset_stats",
    "partitions",
    "curves",
    "thresholds",
    "mining",
    "bars",
]
