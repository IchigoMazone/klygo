import os
from typing import Any, List, Tuple, Union, Optional
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

__all__ = ["draw_bboxes"]

# Official Ultralytics 20-color hex palette, brightened so no dark/black colors exist
ULTRALYTICS_HEX = (
    "042AFF", "0BDBEB", "FF9500", "00DFB7", "3B82F6",
    "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
    "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3",
    "7D24FF", "E02424", "FF1B6C", "FC6D2F", "A2FF0B",
)
ULTRALYTICS_PALETTE = [
    tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in ULTRALYTICS_HEX
]


def _get_palette_color(idx: int) -> tuple:
    """Return a distinct palette color deterministically by class index."""
    return ULTRALYTICS_PALETTE[idx % len(ULTRALYTICS_PALETTE)]


def _find_badge_position(
    box: Tuple[float, float, float, float],
    bw: int,
    bh: int,
    w_img: int,
    h_img: int,
    occupied: List[List[int]],
) -> Tuple[int, int]:
    """Find optimal non-overlapping position for a caption badge near its box."""
    xmin, ymin, xmax, ymax = box
    ixmin = int(round(xmin))
    iymin = int(round(ymin))
    ixmax = int(round(xmax))
    iymax = int(round(ymax))

    bx_base = max(0, min(ixmin, w_img - bw))

    candidates = []

    # 1. Outside top (standard YOLO placement)
    if iymin - bh >= 0:
        candidates.append((bx_base, iymin - bh))

    # 2. Inside top (when near top edge or outside top is occupied)
    if iymin + bh <= h_img:
        candidates.append((bx_base, iymin))

    # 3. Stacked directly above / below colliding occupied badges
    for occ in occupied:
        ox1, oy1, ox2, oy2 = occ
        # Below occupied badge
        if oy2 + bh <= h_img:
            candidates.append((bx_base, oy2))
        # Above occupied badge
        if oy1 - bh >= 0:
            candidates.append((bx_base, oy1 - bh))
        # Horizontal shift side-by-side
        if ox2 + bw <= w_img:
            if iymin - bh >= 0:
                candidates.append((ox2, iymin - bh))
            if iymin + bh <= h_img:
                candidates.append((ox2, iymin))
        if ox1 - bw >= 0:
            if iymin - bh >= 0:
                candidates.append((ox1 - bw, iymin - bh))
            if iymin + bh <= h_img:
                candidates.append((ox1 - bw, iymin))

    # 4. Inside bottom & outside bottom
    if iymax - bh >= 0 and iymax - bh >= iymin:
        candidates.append((bx_base, iymax - bh))
    if iymax + bh <= h_img:
        candidates.append((bx_base, iymax))

    # Evaluate candidates in priority order: first one with 0 overlap wins
    best_candidate = None
    min_overlap = float("inf")

    for cx, cy in candidates:
        cx = max(0, min(cx, w_img - bw))
        cy = max(0, min(cy, h_img - bh))
        cand_rect = [cx, cy, cx + bw, cy + bh]

        total_overlap = 0
        for occ in occupied:
            dx = max(0, min(cand_rect[2], occ[2]) - max(cand_rect[0], occ[0]))
            dy = max(0, min(cand_rect[3], occ[3]) - max(cand_rect[1], occ[1]))
            if dx > 0 and dy > 0:
                total_overlap += dx * dy

        if total_overlap == 0:
            return cx, cy

        if total_overlap < min_overlap:
            min_overlap = total_overlap
            best_candidate = (cx, cy)

    if best_candidate is not None and min_overlap == 0:
        return best_candidate

    return best_candidate if best_candidate is not None else (bx_base, max(0, iymin - bh if iymin - bh >= 0 else iymin))


def draw_bboxes(
    image: Union[PIL.Image.Image, np.ndarray],
    bboxes: Any,
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    thickness: Optional[int] = None,
    line_width: Optional[int] = None,
    font_size: Optional[int] = None,
    box_color: Optional[Tuple[int, int, int]] = None,
    text_color: Optional[Tuple[int, int, int]] = None,
    show_labels: bool = True,
    show_scores: bool = True,
    **kwargs,
) -> Union[PIL.Image.Image, np.ndarray]:
    """Draw bounding boxes and optional captions on a copy of an image matching Ultralytics YOLO.

    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Source image. NumPy images use RGB channel order.
    bboxes : list or Detection
        Boxes in ``[xmin, ymin, xmax, ymax]`` pixel coordinates or a Klygo Detection frame.
    labels : list of str, optional
        Class labels, matched to boxes by position.
    scores : list of float, optional
        Confidence values, matched to boxes by position.
    thickness, line_width : int, optional
        Border width. ``line_width`` takes precedence.
    font_size : int, optional
        Caption font size.
    box_color, text_color : tuple of int, optional
        RGB overrides for the box and caption text (text defaults to crisp white).
    show_labels, show_scores : bool
        Whether to include labels and scores in captions.

    Returns
    -------
    PIL.Image.Image or numpy.ndarray
        Annotated copy in the same representation as ``image``.

    Notes
    -----
    Renders bounding box frames and filled label badges matching Ultralytics YOLO's
    crisp style, vibrant non-black color palette, and pure white text.
    Caption badges automatically avoid overlapping when boxes are close or stacked.
    Extra ``width`` in ``kwargs`` is a legacy border-width fallback.
    """
    # 0. Support Detection frame directly
    if hasattr(bboxes, "boxes") and hasattr(bboxes, "labels"):
        if labels is None:
            labels = list(bboxes.labels)
        if scores is None and hasattr(bboxes, "scores"):
            scores = list(bboxes.scores)
        bboxes = bboxes.boxes

    is_np = isinstance(image, np.ndarray)
    if is_np:
        pil_img = PIL.Image.fromarray(image if image.ndim == 2 or image.shape[2] == 3 else image[:, :, :3])
    elif hasattr(image, "to_pil"):
        pil_img = image.to_pil(cache=False).copy()
    else:
        pil_img = image.copy()

    annotated = pil_img.copy()
    draw = PIL.ImageDraw.Draw(annotated)
    w_img, h_img = annotated.size

    # Normalize bboxes so each box is a 4-float sequence
    norm_boxes = []
    for b in bboxes:
        if hasattr(b, "xmin"):
            norm_boxes.append((float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax)))
        elif hasattr(b, "box"):
            norm_boxes.append(tuple(float(c) for c in b.box))
        else:
            norm_boxes.append(tuple(float(c) for c in b))
    bboxes = norm_boxes

    # 1. Compute dynamic line width and font size (crisp, not overly thick)
    actual_lw = line_width or thickness or kwargs.get("width") or max(round(min(w_img, h_img) * 0.0035), 2)
    actual_fs = font_size or max(round(min(w_img, h_img) * 0.03), 13)

    # 2. Load system TrueType Arial font matching Ultralytics
    ultralytics_font = os.path.join(os.environ.get("APPDATA", ""), "Ultralytics", "Arial.ttf")
    font_candidates = [
        ultralytics_font,
        "Arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
        "calibri.ttf",
        "SegoeUI.ttf",
    ]
    font = None
    for font_name in font_candidates:
        try:
            font = PIL.ImageFont.truetype(font_name, actual_fs)
            break
        except Exception:
            continue
    if font is None:
        font = PIL.ImageFont.load_default()

    # 3. Map unique labels/classes to stable palette colors (no black, vibrant YOLO tones)
    all_labels = labels or []
    def _extract_cls(text: str) -> str:
        parts = text.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[1]
        return text

    classes = list(dict.fromkeys([_extract_cls(str(l)) for l in all_labels]))
    class_to_color = {}
    for i, c in enumerate(classes):
        if box_color is not None:
            class_to_color[c] = box_color
        elif c.isdigit():
            class_to_color[c] = _get_palette_color(int(c))
        else:
            class_to_color[c] = _get_palette_color(i)

    label_to_color = {str(l): class_to_color.get(_extract_cls(str(l)), (4, 42, 255)) for l in all_labels}

    # 4. Pass 1: Render bounding box rectangles with crisp border
    for idx, box in enumerate(bboxes):
        lbl = str(all_labels[idx]) if idx < len(all_labels) else "object"
        color = label_to_color.get(lbl, _get_palette_color(idx) if box_color is None else box_color)
        xmin, ymin, xmax, ymax = map(float, box)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=actual_lw)

    # 5. Pass 2: Render label badges with anti-collision resolution and pure white text
    occupied_badges: List[List[int]] = []
    txt_clr = text_color if text_color is not None else (255, 255, 255)
    pad_x = 3
    pad_y = 2

    for idx, box in enumerate(bboxes):
        lbl = str(all_labels[idx]) if idx < len(all_labels) else ""
        color = label_to_color.get(lbl, _get_palette_color(idx) if box_color is None else box_color)
        xmin, ymin, xmax, ymax = map(float, box)

        text_parts = []
        if show_labels and lbl:
            text_parts.append(lbl)
        if show_scores and scores is not None and idx < len(scores) and scores[idx] is not None:
            text_parts.append(f"{scores[idx]:.2f}")
        caption = " ".join(text_parts)

        if caption:
            tb = font.getbbox(caption)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            bw = tw + 2 * pad_x
            bh = th + 2 * pad_y

            bx, by = _find_badge_position(
                (xmin, ymin, xmax, ymax), bw, bh, w_img, h_img, occupied_badges
            )
            badge_rect = [bx, by, bx + bw, by + bh]
            occupied_badges.append(badge_rect)

            draw.rectangle(badge_rect, fill=color)
            draw.text((bx + pad_x - tb[0], by + pad_y - tb[1]), caption, fill=txt_clr, font=font)

    if is_np:
        return np.array(annotated)
    return annotated
