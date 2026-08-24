import os
from typing import List, Tuple, Union, Optional
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np

# Bảng 20 mã màu Hex chuẩn đẹp của Ultralytics YOLOv8 / YOLOv11
ULTRALYTICS_HEX = (
    "042aff", "0bdfdf", "ff9700", "ff4477", "b644ff", "074799",
    "00a36c", "9e0059", "ff0054", "ff5400", "ffbd00", "2b9348",
    "0077b6", "5a189a", "e0aaff", "7b2cbf", "9d0208", "dc2f02",
    "e85d04", "f48c06"
)
ULTRALYTICS_PALETTE = [
    tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in ULTRALYTICS_HEX
]


def _get_palette_color(idx: int) -> tuple:
    """Lấy màu phân biệt cố định theo index của class từ bảng màu YOLO."""
    return ULTRALYTICS_PALETTE[idx % len(ULTRALYTICS_PALETTE)]


def draw_bboxes(
    image: Union[PIL.Image.Image, np.ndarray],
    bboxes: List[Union[List[float], Tuple[float, float, float, float]]],
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
    """
    Tác dụng:
    - Vẽ bounding box và nhãn tên theo chuẩn đồ họa chuyên nghiệp Ultralytics YOLO (2-Pass Rendering).
    - Tự động phân chia bảng 20 màu độc lập cho từng class.
    - Badge nhãn nổi bật chữ trắng trên nền màu, tự động canh lề và chống đè chéo.

    Đầu vào:
    - image: Ảnh PIL hoặc mảng NumPy
    - bboxes: Danh sách bounding box theo định dạng [xmin, ymin, xmax, ymax]
    - labels: Danh sách nhãn của các bounding box
    - scores: Danh sách điểm tin cậy của các bounding box
    - thickness / line_width: Độ dày nét vẽ viền box
    - font_size: Cỡ chữ nhãn
    - box_color: Màu viền box cố định (nếu muốn override màu mặc định)
    - text_color: Màu chữ nhãn
    - show_labels: Có hiển thị tên nhãn hay không
    - show_scores: Có hiển thị điểm tin cậy hay không

    Đầu ra:
    - Ảnh PIL.Image hoặc NumPy đã vẽ Bounding Box hoàn chỉnh.
    """
    is_np = isinstance(image, np.ndarray)
    if is_np:
        pil_img = PIL.Image.fromarray(image if image.ndim == 2 or image.shape[2] == 3 else image[:, :, :3])
    else:
        pil_img = image.copy()

    annotated = pil_img.copy()
    draw = PIL.ImageDraw.Draw(annotated)
    w_img, h_img = annotated.size

    # 1. Tính toán độ dày viền & cỡ chữ tự động theo tỉ lệ kích thước ảnh
    actual_lw = line_width or thickness or kwargs.get("width") or max(round(sum(annotated.size) / 2 * 0.003), 2)
    actual_fs = font_size or max(round(sum(annotated.size) / 2 * 0.025), 12)

    # 2. Nạp Font chữ hệ thống hỗ trợ UTF-8
    font = None
    for font_name in ["arial.ttf", "DejaVuSans.ttf", "calibri.ttf", "SegoeUI.ttf"]:
        try:
            font = PIL.ImageFont.truetype(font_name, actual_fs)
            break
        except Exception:
            continue
    if font is None:
        font = PIL.ImageFont.load_default()

    # 3. Tạo ánh xạ nhãn -> màu cố định từ bảng màu Ultralytics Palette
    all_labels = labels or []
    unique_labels = list(dict.fromkeys([str(l) for l in all_labels]))
    label_to_color = {
        l: (_get_palette_color(i) if box_color is None else box_color)
        for i, l in enumerate(unique_labels)
    }

    # 4. PASS 1: Vẽ toàn bộ tất cả khung chữ nhật BBox trước
    for idx, box in enumerate(bboxes):
        lbl = str(all_labels[idx]) if idx < len(all_labels) else "object"
        color = label_to_color.get(lbl, (255, 56, 56) if box_color is None else box_color)
        xmin, ymin, xmax, ymax = map(float, box)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=actual_lw)

    # 5. PASS 2: Vẽ toàn bộ Badge nhãn chữ lên trên cùng
    occupied_badges = []
    for idx, box in enumerate(bboxes):
        lbl = str(all_labels[idx]) if idx < len(all_labels) else ""
        color = label_to_color.get(lbl, (255, 56, 56) if box_color is None else box_color)
        xmin, ymin, xmax, ymax = map(float, box)

        text_parts = []
        if show_labels and lbl:
            text_parts.append(lbl)
        if show_scores and scores is not None and idx < len(scores) and scores[idx] is not None:
            text_parts.append(f"{scores[idx]:.2f}")
        caption = " ".join(text_parts)

        if caption:
            tb = draw.textbbox((0, 0), caption, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]

            outside = (ymin - th - 4) >= 0
            b_ymin = (ymin - th - 4) if outside else ymin
            b_ymax = ymin if outside else (ymin + th + 4)
            b_xmin = max(0, xmin)
            b_xmax = min(w_img, b_xmin + tw + 6)
            current_badge = [b_xmin, b_ymin, b_xmax, b_ymax]

            def _overlaps(b1, b2):
                return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

            if any(_overlaps(current_badge, occ) for occ in occupied_badges) and outside:
                b_ymin = ymin
                b_ymax = ymin + th + 4
                current_badge = [b_xmin, b_ymin, b_xmax, b_ymax]

            occupied_badges.append(current_badge)
            draw.rectangle(current_badge, fill=color)
            text_y = b_ymin + 1 if outside and b_ymin == (ymin - th - 4) else b_ymin + 2
            txt_clr = text_color if text_color is not None else (255, 255, 255)
            draw.text((b_xmin + 3, text_y), caption, fill=txt_clr, font=font)

    if is_np:
        return np.array(annotated)
    return annotated
