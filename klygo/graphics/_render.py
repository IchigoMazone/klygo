"""Shared Matplotlib figure finalization and export helpers."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from klygo import files, validators


def save_figure(
    fig: Figure,
    path: str | PathLike[str],
    *,
    dpi: int = 300,
    tight: bool = True,
) -> Path:
    """Save a Matplotlib figure to disk in PNG, PDF, or SVG format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to export.
    path : str or os.PathLike[str]
        Destination file path. Parent directories are created automatically.
    dpi : int, default=300
        Output resolution in dots per inch.
    tight : bool, default=True
        Use tight bounding box when saving.

    Returns
    -------
    pathlib.Path
        Normalized path to the saved figure file.
    """
    validators.validate_type(fig, Figure, "fig")
    validators.validate_type(path, (str, PathLike), "path")
    validators.validate_type(dpi, int, "dpi")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    dest = files.path(path)
    files.mkdir(files.parent(dest))
    fig.savefig(
        str(dest),
        dpi=dpi,
        bbox_inches="tight" if tight else None,
    )
    return dest


def _finalize_figure(
    fig: Figure,
    *,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Save, display, and optionally return a Matplotlib figure."""
    validators.validate_type(show, bool, "show")
    validators.validate_type(return_figure, bool, "return_figure")

    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi, tight=True)
    if show:
        plt.show()
    if return_figure:
        return fig
    plt.close(fig)
    return None


__all__ = ["save_figure", "_finalize_figure"]
