"""Composable grid layout for heterogeneous images, Detections, and Figures."""

from __future__ import annotations

from dataclasses import dataclass, replace
import io
import math
from os import PathLike
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw

from klygo import files, media, outputs, validators

from ._render import _finalize_figure
from .draw_bboxes import draw_bboxes


@dataclass(frozen=True)
class GridCell:
    """Configuration for a single cell inside :func:`grid`.

    Attributes
    ----------
    source : Any
        Image source, path, ``LazyImage``, ``Detection``, or ``Figure``.
    width : int, optional
        Target width in pixels (overrides source width).
    height : int, optional
        Target height in pixels (overrides source height).
    align : str or tuple[int, int], optional
        Placement inside the grid cell (e.g. ``'center'``, ``'top-left'``, or ``(x, y)``).
    offset : tuple[int, int], optional
        Additional ``(dx, dy)`` pixel offset applied after alignment.
    title : str, optional
        Optional caption rendered above the cell.
    """

    source: Any
    width: Optional[int] = None
    height: Optional[int] = None
    align: Optional[str | tuple[int, int]] = None
    offset: Optional[tuple[int, int]] = None
    title: Optional[str] = None


def cell(
    source: Any,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    align: Optional[str | tuple[int, int]] = None,
    offset: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
) -> GridCell:
    """Create a per-cell layout specification for :func:`compose` and :func:`grid`."""
    if width is not None and (isinstance(width, bool) or not isinstance(width, int) or width <= 0):
        raise ValueError("width must be a positive integer or None")
    if height is not None and (isinstance(height, bool) or not isinstance(height, int) or height <= 0):
        raise ValueError("height must be a positive integer or None")
    if offset is not None:
        if not isinstance(offset, (tuple, list)) or len(offset) != 2:
            raise ValueError("offset must be a (dx, dy) tuple of two integers")
        offset = (int(offset[0]), int(offset[1]))
    return GridCell(
        source=source,
        width=width,
        height=height,
        align=align,
        offset=offset,
        title=title,
    )


def _to_pil(source: Any) -> Image.Image:
    """Convert any supported image/detection/figure source into an RGB PIL Image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[2] == 1:
            arr = source[:, :, 0]
        elif source.ndim == 2 or source.shape[2] == 3:
            arr = source
        else:
            arr = source[:, :, :3]
        return Image.fromarray(arr).convert("RGB")
    if isinstance(source, (str, PathLike)):
        return media.load(files.path(source), verbose=False)[0].to_pil(cache=False).convert("RGB")
    if isinstance(source, outputs.Detection):
        return draw_bboxes(source.image, source).convert("RGB")
    if isinstance(source, Figure):
        buf = io.BytesIO()
        source.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        buf.close()
        return img
    if hasattr(source, "to_pil"):
        return source.to_pil(cache=False).convert("RGB")
    raise TypeError(f"Unsupported grid cell source type: {type(source).__name__}")


def _resize_cell_image(
    img: Image.Image,
    width: Optional[int],
    height: Optional[int],
) -> Image.Image:
    """Resize a PIL image honoring explicit width/height overrides."""
    w0, h0 = img.size
    if width is None and height is None:
        return img
    if width is not None and height is not None:
        target_w, target_h = width, height
    elif width is not None:
        target_w = width
        target_h = max(1, int(round(h0 * (width / max(1, w0)))))
    else:
        assert height is not None
        target_h = height
        target_w = max(1, int(round(w0 * (height / max(1, h0)))))
    if (target_w, target_h) == (w0, h0):
        return img
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def compose(cells: Sequence[GridCell]) -> list[GridCell]:
    """Normalize a sequence of :func:`cell` items into resolved RGB images.

    Parameters
    ----------
    cells : Sequence[GridCell]
        Sequence where every item is created by :func:`cell`.

    Returns
    -------
    list[GridCell]
        Normalized ``GridCell`` list with resolved and resized ``PIL.Image.Image`` sources.
    """
    validators.validate_type(cells, (list, tuple), "cells")
    if not cells:
        raise ValueError("cells must not be empty")

    resolved: list[GridCell] = []
    for idx, item in enumerate(cells):
        if not isinstance(item, GridCell):
            raise TypeError(f"compose expects GridCell items; got {type(item).__name__} at index {idx}")
        pil_img = _to_pil(item.source)
        resized = _resize_cell_image(pil_img, item.width, item.height)
        resolved.append(replace(item, source=resized))
    return resolved


def _resolve_position(
    cell_w: int,
    cell_h: int,
    img_w: int,
    img_h: int,
    align: Optional[str | tuple[int, int]],
    offset: Optional[tuple[int, int]],
) -> tuple[int, int]:
    """Compute (x, y) top-left paste coordinate inside a grid cell."""
    dx, dy = offset or (0, 0)
    if isinstance(align, (tuple, list)):
        if len(align) != 2:
            raise ValueError("align tuple must have 2 coordinates (x, y)")
        return int(align[0]) + dx, int(align[1]) + dy

    anchor = (align or "center").lower()
    anchors = {
        "center": ((cell_w - img_w) // 2, (cell_h - img_h) // 2),
        "top": ((cell_w - img_w) // 2, 0),
        "bottom": ((cell_w - img_w) // 2, cell_h - img_h),
        "left": (0, (cell_h - img_h) // 2),
        "right": (cell_w - img_w, (cell_h - img_h) // 2),
        "top-left": (0, 0),
        "top-right": (cell_w - img_w, 0),
        "bottom-left": (0, cell_h - img_h),
        "bottom-right": (cell_w - img_w, cell_h - img_h),
    }
    if anchor not in anchors:
        raise ValueError(f"Unsupported align value: {align!r}")
    x, y = anchors[anchor]
    return x + dx, y + dy


def _normalize_padding(padding: int | tuple[int, ...]) -> tuple[int, int, int, int]:
    """Normalize padding into (top, right, bottom, left)."""
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    if len(padding) == 2:
        return (int(padding[0]), int(padding[1]), int(padding[0]), int(padding[1]))
    if len(padding) == 4:
        return (int(padding[0]), int(padding[1]), int(padding[2]), int(padding[3]))
    raise ValueError("padding must be an int or a 2-tuple / 4-tuple of ints")


def grid(
    items: Sequence[Any],
    *,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    cell_size: Optional[tuple[int, int]] = None,
    gap: int | tuple[int, int] = 16,
    padding: int | tuple[int, ...] = 24,
    bg_color: str | tuple[int, int, int] = "white",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Arrange images or :func:`compose` cells into a ``rows x cols`` grid figure.

    Parameters
    ----------
    items : Sequence[GridCell or image source]
        Either normalized ``GridCell`` items from :func:`compose` or plain image sources.
    rows, cols : int, optional
        Grid dimensions. When omitted, defaults to a single row or computed from the other.
    cell_size : tuple[int, int], optional
        Fixed ``(width, height)`` in pixels for every cell. Defaults to the max item size.
    gap : int or tuple[int, int], default=16
        Spacing between cells in pixels ``(gap_x, gap_y)``.
    padding : int or tuple[int, ...], default=24
        Outer canvas padding ``(top, right, bottom, left)`` in pixels.
    bg_color : str or tuple[int, int, int], default='white'
        Background color for the grid canvas.
    title : str, optional
        Overall title displayed above the grid.
    show : bool, default=True
        Display the rendered grid.
    save_path : str or os.PathLike[str], optional
        Optional output path to save the grid image/PDF.
    dpi : int, default=300
        Output resolution when saving.
    return_figure : bool, default=False
        Return the underlying Matplotlib ``Figure`` instead of closing it.
    """
    validators.validate_type(items, (list, tuple), "items")
    if not items:
        raise ValueError("items must not be empty")

    cells: list[GridCell] = [
        item if isinstance(item, GridCell) else cell(item)
        for item in items
    ]
    resolved = compose(cells)

    n = len(resolved)
    if cols is None and rows is None:
        cols_val = n
        rows_val = 1
    elif cols is not None and rows is None:
        cols_val = int(cols)
        rows_val = math.ceil(n / cols_val)
    elif rows is not None and cols is None:
        rows_val = int(rows)
        cols_val = math.ceil(n / rows_val)
    else:
        assert rows is not None and cols is not None
        rows_val, cols_val = int(rows), int(cols)

    if rows_val <= 0 or cols_val <= 0 or rows_val * cols_val < n:
        raise ValueError("rows * cols must be positive and >= len(items)")

    gap_x, gap_y = (gap, gap) if isinstance(gap, int) else (int(gap[0]), int(gap[1]))
    pad_t, pad_r, pad_b, pad_l = _normalize_padding(padding)

    if cell_size is not None:
        cell_w, cell_h = int(cell_size[0]), int(cell_size[1])
    else:
        cell_w = max(item.source.width for item in resolved)
        cell_h = max(item.source.height for item in resolved)

    caption_h = 26 if any(item.title for item in resolved) else 0
    total_w = pad_l + cols_val * cell_w + max(0, cols_val - 1) * gap_x + pad_r
    total_h = pad_t + rows_val * (cell_h + caption_h) + max(0, rows_val - 1) * gap_y + pad_b

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(resolved):
        r = idx // cols_val
        c = idx % cols_val
        origin_x = pad_l + c * (cell_w + gap_x)
        origin_y = pad_t + r * (cell_h + caption_h + gap_y)

        if item.title:
            draw.text((origin_x + 4, origin_y + 4), item.title, fill=(20, 20, 20))

        img = item.source
        px, py = _resolve_position(cell_w, cell_h, img.width, img.height, item.align, item.offset)
        canvas.paste(img, (origin_x + px, origin_y + caption_h + py))

    fig, ax = plt.subplots(figsize=(max(4.0, total_w / 100.0), max(3.0, total_h / 100.0)), dpi=100)
    ax.imshow(np.asarray(canvas))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout(pad=0.2)

    return _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        dpi=dpi,
        return_figure=return_figure,
    )


__all__ = ["GridCell", "cell", "compose", "grid"]
