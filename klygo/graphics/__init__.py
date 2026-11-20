"""Non-destructive visualization, adaptive display, composable grids, and report charts.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/graphics/Document_Graphics_Overview.ipynb

Public API (13 functions):
    Image rendering & display:
        1. draw_bboxes(...)                 Render bounding boxes and labels onto an image copy.
        2. show_image(...)                  Display an image in notebook, GUI, or terminal environments.
        3. compare_images(...)              Render side-by-side Detection comparisons.
        4. plot_dataset_stats(...)          Plot class distribution and object counts.

    Composable grid layout:
        5. cell(...)                        Configure a single cell for compose() and grid().
        6. compose(...)                     Normalize heterogeneous cell sources into images.
        7. grid(...)                        Arrange cells or images into a rows x cols grid.

    Report charts & export:
        8. plot_partition_distribution(...) Plot image counts and percentages per split.
        9. plot_loss_curves(...)            Plot training or validation loss/metric curves.
       10. plot_threshold_sweep(...)        Plot P/R/F1 curves or TP/FP counts by threshold.
       11. plot_mining_summary(...)         Plot mining positive/background or confidence bars.
       12. plot_benchmark_bars(...)         Plot single-metric (± std) or multi-metric bars.
       13. save_figure(...)                 Export a Matplotlib figure to PNG, PDF, or SVG.
"""

from .draw_bboxes import draw_bboxes, draw_box
from .plot_dataset_stats import plot_dataset_stats
from .show_image import show_image
from .compare_images import compare_images
from .layout import cell, compose, grid
from .charts import (
    plot_partition_distribution,
    plot_loss_curves,
    plot_threshold_sweep,
    plot_mining_summary,
    plot_benchmark_bars,
)
from ._render import save_figure

__all__ = [
    "draw_bboxes",
    "plot_dataset_stats",
    "show_image",
    "compare_images",
    "cell",
    "compose",
    "grid",
    "plot_partition_distribution",
    "plot_loss_curves",
    "plot_threshold_sweep",
    "plot_mining_summary",
    "plot_benchmark_bars",
    "save_figure",
]
