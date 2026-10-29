"""Two-pass non-destructive bounding box rendering matching Ultralytics YOLO style."""

import os
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

__all__ = ["draw_bboxes"]

# Official Ultralytics 20-color hex palette for fallback rendering
ULTRALYTICS_HEX = (
    "042AFF", "0BDBEB", "FF9500", "00DFB7", "3B82F6",
    "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
    "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3",
    "7D24FF", "E02424", "FF1B6C", "FC6D2F", "A2FF0B",
)
ULTRALYTICS_PALETTE = [
    tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in ULTRALYTICS_HEX
]


def _get_annotator():
    """Lazily load Ultralytics Annotator and colors to avoid eager framework imports."""
    try:
        from ultralytics.utils.plotting import Annotator, colors
        return Annotator, colors
    except ImportError:
        return None, None


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

    Leverages Ultralytics ``Annotator`` and ``colors`` to deliver native, publication-grade
    YOLO bounding box outlines, vibrant deterministic color palettes, and crisp label badges.

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
        working_np = base_rgb.copy()
    elif hasattr(image, "to_pil"):
        pil_source = image.to_pil(cache=False)
        working_np = np.asarray(pil_source.convert("RGB")).copy()
    elif isinstance(image, PIL.Image.Image):
        working_np = np.asarray(image.convert("RGB")).copy()
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected PIL.Image.Image, numpy.ndarray, or LazyImage."
        )

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

    # 3. Determine line width
    actual_lw = line_width or thickness or kwargs.get("width")
    if actual_lw is not None:
        actual_lw = int(actual_lw)

    # 4. Initialize Ultralytics Annotator lazily
    Annotator, colors = _get_annotator()

    if Annotator is not None and colors is not None:
        annotator = Annotator(
            working_np,
            line_width=actual_lw,
            font_size=font_size,
            pil=True,
        )

        all_labels = labels or []
        unique_labels = list(dict.fromkeys([str(lbl) for lbl in all_labels]))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        txt_clr = text_color if text_color is not None else (255, 255, 255)

        for idx, box in enumerate(bboxes):
            lbl = str(all_labels[idx]) if idx < len(all_labels) and all_labels[idx] is not None else ""
            text_parts = []
            if show_labels and lbl:
                text_parts.append(lbl)
            if show_scores and scores is not None and idx < len(scores) and scores[idx] is not None:
                text_parts.append(f"{scores[idx]:.2f}")
            caption = " ".join(text_parts) if text_parts else None

            if box_color is not None:
                c = box_color
            elif class_ids is not None and idx < len(class_ids) and class_ids[idx] is not None:
                c = colors(int(class_ids[idx]), True)
            elif lbl:
                parts = lbl.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    c = colors(int(parts[0]), True)
                elif lbl.isdigit():
                    c = colors(int(lbl), True)
                else:
                    c = colors(label_to_idx.get(lbl, idx), True)
            else:
                c = colors(idx, True)

            annotator.box_label(box, label=caption, color=c, txt_color=txt_clr)

        annotated_np = annotator.result()
    else:
        # Fallback to direct PIL rendering
        annotated_pil = PIL.Image.fromarray(working_np)
        draw = PIL.ImageDraw.Draw(annotated_pil)
        lw = actual_lw or max(round(min(working_np.shape[:2]) * 0.0035), 2)
        fs = font_size or max(round(min(working_np.shape[:2]) * 0.03), 13)
        try:
            font = PIL.ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = PIL.ImageFont.load_default()

        all_labels = labels or []
        for idx, box in enumerate(bboxes):
            c = box_color or ULTRALYTICS_PALETTE[idx % len(ULTRALYTICS_PALETTE)]
            draw.rectangle(box, outline=c, width=lw)
            lbl = str(all_labels[idx]) if idx < len(all_labels) else ""
            text_parts = []
            if show_labels and lbl:
                text_parts.append(lbl)
            if show_scores and scores is not None and idx < len(scores) and scores[idx] is not None:
                text_parts.append(f"{scores[idx]:.2f}")
            caption = " ".join(text_parts)
            if caption:
                tb = font.getbbox(caption)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                draw.rectangle([box[0], max(0, box[1] - th - 4), box[0] + tw + 6, max(th + 4, box[1])], fill=c)
                draw.text((box[0] + 3, max(0, box[1] - th - 2)), caption, fill=(255, 255, 255), font=font)
        annotated_np = np.asarray(annotated_pil)

    # 5. Format return value matching input type
    if is_np:
        if has_alpha and alpha_channel is not None:
            return np.concatenate([annotated_np, alpha_channel], axis=-1)
        return annotated_np

    return PIL.Image.fromarray(annotated_np)


# Convenient alias
draw_box = draw_bboxes
