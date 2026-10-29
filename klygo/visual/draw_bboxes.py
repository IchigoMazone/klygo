"""Two-pass non-destructive bounding box rendering matching Ultralytics YOLO style with anti-collision labels."""

import os
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

__all__ = ["draw_bboxes"]

# Official Ultralytics 20-color hex palette
ULTRALYTICS_HEX = (
    "042AFF", "0BDBEB", "FF9500", "00DFB7", "3B82F6",
    "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
    "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3",
    "7D24FF", "E02424", "FF1B6C", "FC6D2F", "A2FF0B",
)
ULTRALYTICS_PALETTE = [
    tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in ULTRALYTICS_HEX
]


def _get_yolo_colors():
    """Lazily load Ultralytics colors object if available."""
    try:
        from ultralytics.utils.plotting import colors
        return colors
    except ImportError:
        return None


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
    """Draw bounding boxes and optional caption badges on a copy of an image matching Ultralytics YOLO.

    Leverages Ultralytics ``colors`` and two-pass anti-collision badge positioning to deliver
    crisp YOLO outlines without overlapping or obscured label text.

    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Source image. NumPy images must use RGB channel order.
    bboxes : list or Detection
        Boxes in ``[xmin, ymin, xmax, ymax]`` pixel coordinates or a Klygo Detection frame.
    labels : list of str, optional
        Class labels, matched to boxes by index.
    scores : list of float, optional
        Confidence values, matched to boxes by index.
    thickness, line_width : int, optional
        Border width in pixels. ``line_width`` takes precedence.
    font_size : int, optional
        Caption font size in pixels.
    box_color, text_color : tuple of int, optional
        RGB overrides for the box border and caption text (text defaults to crisp white).
    show_labels, show_scores : bool
        Whether to include labels and scores in captions.
    **kwargs : Any
        Supports legacy ``width`` keyword as a fallback border width.

    Returns
    -------
    PIL.Image.Image or numpy.ndarray
        Annotated copy in the same representation as ``image``.
    """
    class_ids = None

    # 0. Unpack Klygo Detection container if passed
    if hasattr(bboxes, "boxes"):
        if labels is None and hasattr(bboxes, "labels"):
            labels = list(bboxes.labels)
        if scores is None and hasattr(bboxes, "scores"):
            scores = list(bboxes.scores)
        if hasattr(bboxes, "class_ids"):
            class_ids = list(bboxes.class_ids)
        bboxes = bboxes.boxes

    # 1. Normalize input image representation without modifying original
    if hasattr(image, "__len__") and not isinstance(image, (np.ndarray, PIL.Image.Image, str, bytes)) and hasattr(image, "__getitem__"):
        if len(image) > 0:
            image = image[0]

    is_np = isinstance(image, np.ndarray)
    has_alpha = False
    alpha_channel = None

    if is_np:
        if image.ndim == 2:
            base_rgb = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            has_alpha = True
            alpha_channel = image[:, :, 3:4]
            base_rgb = image[:, :, :3]
        elif image.ndim == 3 and image.shape[2] >= 3:
            base_rgb = image[:, :, :3]
        else:
            base_rgb = image
        working_img = PIL.Image.fromarray(base_rgb)
    elif hasattr(image, "to_pil"):
        working_img = image.to_pil(cache=False).copy()
    elif isinstance(image, PIL.Image.Image):
        working_img = image.copy()
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected PIL.Image.Image, numpy.ndarray, or LazyImage."
        )

    annotated = working_img.copy()
    draw = PIL.ImageDraw.Draw(annotated)
    w_img, h_img = annotated.size

    # 2. Normalize bboxes into sequences of 4-float tuples [xmin, ymin, xmax, ymax]
    norm_boxes = []
    for b in bboxes:
        if hasattr(b, "xmin"):
            norm_boxes.append((float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax)))
        elif hasattr(b, "box"):
            norm_boxes.append(tuple(float(c) for c in b.box))
        else:
            norm_boxes.append(tuple(float(c) for c in b))
    bboxes = norm_boxes

    # 3. Dynamic line width and font size matching Ultralytics
    actual_lw = line_width or thickness or kwargs.get("width") or max(round(min(w_img, h_img) * 0.0035), 2)
    actual_fs = font_size or max(round(min(w_img, h_img) * 0.03), 13)

    # 4. Load TrueType font matching Ultralytics Arial
    ultralytics_font = os.path.join(os.environ.get("APPDATA", ""), "Ultralytics", "Arial.ttf")
    font = None
    for font_name in [ultralytics_font, "Arial.ttf", "arial.ttf", "DejaVuSans.ttf", "calibri.ttf", "SegoeUI.ttf"]:
        try:
            font = PIL.ImageFont.truetype(font_name, actual_fs)
            break
        except Exception:
            continue
    if font is None:
        font = PIL.ImageFont.load_default()

    # 5. Resolve official YOLO palette colors
    yolo_colors = _get_yolo_colors()
    all_labels = labels or []
    unique_labels = list(dict.fromkeys([str(lbl) for lbl in all_labels]))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    def _resolve_color(idx: int) -> Tuple[int, int, int]:
        if box_color is not None:
            return box_color
        if class_ids is not None and idx < len(class_ids) and class_ids[idx] is not None:
            cid = int(class_ids[idx])
            return yolo_colors(cid, True) if yolo_colors else ULTRALYTICS_PALETTE[cid % len(ULTRALYTICS_PALETTE)]
        if idx < len(all_labels) and all_labels[idx] is not None:
            lbl = str(all_labels[idx])
            parts = lbl.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                cid = int(parts[0])
                return yolo_colors(cid, True) if yolo_colors else ULTRALYTICS_PALETTE[cid % len(ULTRALYTICS_PALETTE)]
            if lbl.isdigit():
                cid = int(lbl)
                return yolo_colors(cid, True) if yolo_colors else ULTRALYTICS_PALETTE[cid % len(ULTRALYTICS_PALETTE)]
            c_idx = label_to_idx.get(lbl, idx)
            return yolo_colors(c_idx, True) if yolo_colors else ULTRALYTICS_PALETTE[c_idx % len(ULTRALYTICS_PALETTE)]
        return yolo_colors(idx, True) if yolo_colors else ULTRALYTICS_PALETTE[idx % len(ULTRALYTICS_PALETTE)]

    box_color_list = [_resolve_color(i) for i in range(len(bboxes))]

    # 6. Pass 1: Render bounding box rectangles with crisp borders
    for idx, box in enumerate(bboxes):
        c = box_color_list[idx]
        xmin, ymin, xmax, ymax = map(float, box)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=c, width=actual_lw)

    # 7. Pass 2: Render label badges with anti-collision resolution so no label is covered
    occupied_badges: List[List[int]] = []
    txt_clr = text_color if text_color is not None else (255, 255, 255)
    pad_x = 3
    pad_y = 2

    for idx, box in enumerate(bboxes):
        c = box_color_list[idx]
        lbl = str(all_labels[idx]) if idx < len(all_labels) and all_labels[idx] is not None else ""
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

            bx, by = _find_badge_position(box, bw, bh, w_img, h_img, occupied_badges)
            badge_rect = [bx, by, bx + bw, by + bh]
            occupied_badges.append(badge_rect)

            draw.rectangle(badge_rect, fill=c)
            draw.text((bx + pad_x - tb[0], by + pad_y - tb[1]), caption, fill=txt_clr, font=font)

    # 8. Format return value matching input type
    if is_np:
        res_np = np.asarray(annotated)
        if has_alpha and alpha_channel is not None:
            return np.concatenate([res_np, alpha_channel], axis=-1)
        return res_np

    return annotated


# Convenient alias
draw_box = draw_bboxes
