import os
from typing import List, Dict, Any, Optional, Union
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from klygo import files, media

# Bảng 20 mã màu Hex chuẩn của Ultralytics YOLOv8 / YOLOv11
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


def _is_notebook() -> bool:
    """Kiểm tra chính xác runtime có thực sự là Jupyter Notebook / Google Colab / Kaggle hay không."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        shell_name = ip.__class__.__name__
        if shell_name in ("ZMQInteractiveShell", "Shell"):
            return True
        if "google.colab" in str(type(ip)):
            return True
        return False
    except Exception:
        return False


class DetectedObject:
    """
    Đại diện cho một vật thể đơn lẻ được nhận diện trong ảnh.
    """

    def __init__(
        self,
        label: str,
        score: float,
        box: List[float],
        img_size: Optional[tuple] = None,
    ) -> None:
        self.label = label
        self.score = score
        self.box = box  # [xmin, ymin, xmax, ymax]
        self.img_size = img_size  # (width, height)

    @property
    def xmin(self) -> float:
        """Tọa độ pixel cạnh trái."""
        return self.box[0]

    @property
    def ymin(self) -> float:
        """Tọa độ pixel cạnh trên."""
        return self.box[1]

    @property
    def xmax(self) -> float:
        """Tọa độ pixel cạnh phải."""
        return self.box[2]

    @property
    def ymax(self) -> float:
        """Tọa độ pixel cạnh dưới."""
        return self.box[3]

    @property
    def box_normalized(self) -> List[float]:
        """Tọa độ chuẩn hóa theo tỉ lệ [0.0, 1.0] dạng [xmin, ymin, xmax, ymax]."""
        if self.img_size and self.img_size[0] > 0 and self.img_size[1] > 0:
            w, h = self.img_size
            return [
                round(self.box[0] / w, 4),
                round(self.box[1] / h, 4),
                round(self.box[2] / w, 4),
                round(self.box[3] / h, 4),
            ]
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin vật thể sang dạng Dictionary."""
        return {
            "label": self.label,
            "score": self.score,
            "box": self.box,
            "box_normalized": self.box_normalized,
        }

    def __repr__(self) -> str:
        return f"DetectedObject(label='{self.label}', score={self.score:.3f}, box={self.box})"


class DetectionResult:
    """
    Đầu ra chuẩn hóa chứa tất cả kết quả nhận diện của một bức ảnh.
    """

    def __init__(
        self,
        source_image: PIL.Image.Image,
        objects: List[DetectedObject],
        speed: Optional[dict] = None,
    ) -> None:
        self.source_image = source_image
        self.objects = objects
        self.speed = speed or {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    @property
    def boxes(self) -> List[List[float]]:
        """Trả về danh sách tọa độ pixel tuyệt đối [xmin, ymin, xmax, ymax]."""
        return [obj.box for obj in self.objects]

    @property
    def normalized_boxes(self) -> List[List[float]]:
        """Trả về danh sách tọa độ chuẩn hóa tỉ lệ trong khoảng [0.0, 1.0]."""
        w, h = self.source_image.width, self.source_image.height
        if w > 0 and h > 0:
            return [
                [
                    round(box[0] / w, 4),
                    round(box[1] / h, 4),
                    round(box[2] / w, 4),
                    round(box[3] / h, 4),
                ]
                for box in self.boxes
            ]
        return []

    @property
    def labels(self) -> List[str]:
        """Trả về danh sách các nhãn tên lớp."""
        return [obj.label for obj in self.objects]

    @property
    def scores(self) -> List[float]:
        """Trả về danh sách các điểm số tin cậy."""
        return [obj.score for obj in self.objects]

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi toàn bộ kết quả về dạng dict chứa trực tiếp boxes, normalized_boxes, labels, scores."""
        return {
            "boxes": self.boxes,
            "normalized_boxes": self.normalized_boxes,
            "labels": self.labels,
            "scores": self.scores,
            "speed": self.speed,
        }

    def plot(
        self,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        scores: bool = True,
        boxes: bool = True,
        outline_color: Optional[str] = None,
        width: Optional[int] = None,
        conf: Optional[bool] = None,
        score: Optional[bool] = None,
    ) -> PIL.Image.Image:
        """
        Tác dụng:
        - Vẽ bounding box và nhãn tên theo chuẩn đồ họa nguyên bản của Ultralytics YOLO.

        Đầu vào:
        - line_width [int]: Độ dày viền khung (Mặc định: tự động tính theo tỉ lệ kích thước ảnh).
        - font_size [int]: Cỡ chữ của nhãn (Mặc định: tự động tính theo tỉ lệ kích thước ảnh).
        - labels [bool]: Hiển thị tên nhãn lớp. Mặc định: True.
        - scores [bool]: Hiển thị điểm số tin cậy (Confidence / Score). Mặc định: True.
        - boxes [bool]: Vẽ khung chữ nhật BBox. Mặc định: True.
        - outline_color [str]: Màu viền tùy chỉnh đơn lẻ (nếu không truyền sẽ dùng bảng 20 màu YOLO phân biệt theo Class).
        - width [int]: Tham số tương thích ngược (alias của line_width).
        - conf [bool]: Tham số tương thích ngược (alias của scores).
        - score [bool]: Tham số tương thích ngược (alias của scores).

        Đầu ra:
        - [PIL.Image.Image]: Bức ảnh mới đã được vẽ đóng khung và gắn tag nhãn.
        """
        # Hỗ trợ linh hoạt cả scores, score và conf
        show_scores = scores
        if score is not None:
            show_scores = score
        elif conf is not None:
            show_scores = conf

        annotated = self.source_image.copy()
        draw = PIL.ImageDraw.Draw(annotated)
        w_img, h_img = self.source_image.size

        # 1. Tính toán độ dày viền & cỡ chữ tự động theo tỉ lệ kích thước ảnh (chuẩn YOLO)
        actual_lw = line_width or width or max(round(sum(self.source_image.size) / 2 * 0.003), 2)
        actual_fs = font_size or max(round(sum(self.source_image.size) / 2 * 0.025), 12)

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

        # 3. Tạo ánh xạ nhãn -> màu cố định
        unique_labels = list(dict.fromkeys(self.labels))
        label_to_color = {
            l: (_get_palette_color(i) if outline_color is None else outline_color)
            for i, l in enumerate(unique_labels)
        }

        # 4. PASS 1: Vẽ toàn bộ tất cả khung chữ nhật BBox trước
        if boxes:
            for obj in self.objects:
                color = label_to_color.get(obj.label, (255, 56, 56))
                xmin, ymin, xmax, ymax = obj.box
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=actual_lw)

        # 5. PASS 2: Vẽ toàn bộ Badge nhãn chữ lên trên cùng (100% không bao giờ bị viền khung cắt đè)
        occupied_badges = []
        for obj in self.objects:
            color = label_to_color.get(obj.label, (255, 56, 56))
            xmin, ymin, xmax, ymax = obj.box

            text_parts = []
            if labels:
                text_parts.append(str(obj.label))
            if show_scores and obj.score is not None:
                text_parts.append(f"{obj.score:.2f}")
            caption = " ".join(text_parts)

            if caption:
                tb = draw.textbbox((0, 0), caption, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]

                # Ưu tiên đặt badge ở phía trên ngoài box (outside)
                outside = (ymin - th - 4) >= 0
                b_ymin = (ymin - th - 4) if outside else ymin
                b_ymax = ymin if outside else (ymin + th + 4)
                b_xmin = max(0, xmin)
                b_xmax = min(w_img, b_xmin + tw + 6)
                current_badge = [b_xmin, b_ymin, b_xmax, b_ymax]

                # Tránh va chạm/đè nhãn: Nếu góc ngoài đã bị nhãn khác chiếm, tự động chuyển vào trong (inside)
                def _overlaps(b1, b2):
                    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

                if any(_overlaps(current_badge, occ) for occ in occupied_badges) and outside:
                    # Lật nhãn vào trong box để không bị chồng lên nhãn của quả cam bên cạnh
                    b_ymin = ymin
                    b_ymax = ymin + th + 4
                    current_badge = [b_xmin, b_ymin, b_xmax, b_ymax]

                occupied_badges.append(current_badge)

                # Vẽ nền Badge cùng màu viền BBox
                draw.rectangle(current_badge, fill=color)

                # Vẽ chữ màu trắng sắc nét
                text_y = b_ymin + 1 if outside and b_ymin == (ymin - th - 4) else b_ymin + 2
                draw.text((b_xmin + 3, text_y), caption, fill=(255, 255, 255), font=font)

        return annotated

    def save(
        self,
        output_path: str,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        scores: bool = True,
        boxes: bool = True,
        outline_color: Optional[str] = None,
        width: Optional[int] = None,
        conf: Optional[bool] = None,
        score: Optional[bool] = None,
    ) -> None:
        """Vẽ và lưu trực tiếp ảnh nhận diện ra đường dẫn đĩa bằng klygo.media.save."""
        annotated = self.plot(
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            scores=scores,
            boxes=boxes,
            outline_color=outline_color,
            width=width,
            conf=conf,
            score=score,
        )
        media.save(output_path, annotated, overwrite=True, verbose=False)

    def show(
        self,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        scores: bool = True,
        boxes: bool = True,
        outline_color: Optional[str] = None,
        width: Optional[int] = None,
        conf: Optional[bool] = None,
        score: Optional[bool] = None,
    ) -> None:
        """
        Vẽ và hiển thị ảnh nhận diện (tương thích cả Notebook lẫn Desktop):
        - Trên Google Colab / Jupyter Notebook: Hiển thị 1 ảnh inline duy nhất dưới cell output.
        - Trên Desktop IDE (VSCode, PyCharm, Terminal): Mở cửa sổ xem ảnh hệ điều hành.
        """
        annotated = self.plot(
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            scores=scores,
            boxes=boxes,
            outline_color=outline_color,
            width=width,
            conf=conf,
            score=score,
        )
        if _is_notebook():
            try:
                from IPython.display import display
                display(annotated)
            except Exception:
                annotated.show()
        else:
            annotated.show()

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def __repr__(self) -> str:
        summary = f"DetectionResult: {len(self.objects)} objects detected"
        if len(self.objects) > 0:
            details = ", ".join([f"{obj.label} ({obj.score:.2f})" for obj in self.objects[:5]])
            if len(self.objects) > 5:
                details += ", ..."
            summary += f" [{details}]"
        return summary


class CroppedObject:
    """
    Đại diện cho một ảnh con đã được cắt kèm siêu dữ liệu nhãn, độ tin cậy và tọa độ gốc.
    """

    def __init__(
        self,
        image: PIL.Image.Image,
        label: str,
        score: float,
        box: List[float],
        box_normalized: Optional[List[float]] = None,
    ) -> None:
        self.image = image
        self.label = label
        self.score = score
        self.box = box  # [xmin, ymin, xmax, ymax]
        self.box_normalized = box_normalized or []

    def __array__(self, dtype=None):
        """Hỗ trợ tự động chuyển đổi sang NumPy Array và tương thích trực tiếp với matplotlib.pyplot.imshow(crop)."""
        import numpy as np
        arr = np.asarray(self.image)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __getattr__(self, name: str) -> Any:
        """Chuyển tiếp các thuộc tính/hàm của PIL Image (như .size, .resize, .convert) sang self.image."""
        return getattr(self.image, name)

    def show(self, width: Optional[int] = None) -> None:
        """
        Mở xem ảnh con (tương thích cả Notebook lẫn Desktop).

        Đầu vào:
        - width [int]: Chiều rộng hiển thị (tùy chọn phóng to/thu nhỏ khi xem trên Colab/Jupyter).
        """
        img_to_show = self.image
        if width is not None and width > 0 and self.image.width > 0:
            new_h = max(1, int(self.image.height * width / self.image.width))
            img_to_show = self.image.resize((width, new_h), PIL.Image.Resampling.BILINEAR)

        if _is_notebook():
            try:
                from IPython.display import display
                display(img_to_show)
            except Exception:
                img_to_show.show()
        else:
            img_to_show.show()

    def save(self, output_path: str) -> None:
        """Lưu ảnh con ra đường dẫn file bằng klygo.media.save."""
        media.save(output_path, self.image, overwrite=True, verbose=False)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin ảnh con sang dạng Dictionary."""
        return {
            "label": self.label,
            "score": self.score,
            "box": self.box,
            "box_normalized": self.box_normalized,
            "size": self.image.size,
        }

    def __repr__(self) -> str:
        return f"CroppedObject(label='{self.label}', score={self.score:.3f}, size={self.image.size})"


class CropResult:
    """
    Đầu ra chuẩn hóa chứa danh sách tất cả các ảnh con được cắt từ một bức ảnh.
    """

    def __init__(self, source_image: PIL.Image.Image, crops: List[CroppedObject]) -> None:
        self.source_image = source_image
        self.crops = crops

    @property
    def images(self) -> List[PIL.Image.Image]:
        """Trả về danh sách đối tượng ảnh PIL.Image thuần túy."""
        return [crop.image for crop in self.crops]

    @property
    def labels(self) -> List[str]:
        """Trả về danh sách tên nhãn của các ảnh con."""
        return [crop.label for crop in self.crops]

    @property
    def scores(self) -> List[float]:
        """Trả về danh sách điểm tin cậy của các ảnh con."""
        return [crop.score for crop in self.crops]

    @property
    def boxes(self) -> List[List[float]]:
        """Trả về danh sách tọa độ gốc của các ảnh con."""
        return [crop.box for crop in self.crops]

    def __len__(self) -> int:
        return len(self.crops)

    def __iter__(self):
        return iter(self.crops)

    def __getitem__(self, index):
        return self.crops[index]

    def show(self, width: Optional[int] = None) -> None:
        """
        Mở xem tất cả các ảnh con.

        Đầu vào:
        - width [int]: Chiều rộng hiển thị (tùy chọn).
        """
        for crop in self.crops:
            crop.show(width=width)

    def save(self, output_dir: str, prefix: str = "crop") -> List[str]:
        """
        Tác dụng:
        - Tự động lưu toàn bộ danh sách ảnh con vào một thư mục được chỉ định.

        Đầu vào:
        - output_dir [str]: Đường dẫn thư mục đích.
        - prefix [str]: Tiền tố đặt tên file (Mặc định: 'crop').

        Đầu ra:
        - [List[str]]: Danh sách các đường dẫn file đã lưu thành công.
        """
        files.mkdir(output_dir)
        saved_paths = []
        for i, crop in enumerate(self.crops, 1):
            file_name = f"{prefix}_{i}_{crop.label}_{crop.score:.2f}.jpg"
            file_path = os.path.join(output_dir, file_name)
            media.save(file_path, crop.image, overwrite=True, verbose=False)
            saved_paths.append(file_path)
        return saved_paths

    def to_dict(self) -> List[Dict[str, Any]]:
        """Chuyển đổi danh sách ảnh con sang list of dicts."""
        return [crop.to_dict() for crop in self.crops]

    def __len__(self) -> int:
        return len(self.crops)

    def __iter__(self):
        return iter(self.crops)

    def __getitem__(self, index):
        return self.crops[index]

    def __repr__(self) -> str:
        summary = f"CropResult: {len(self.crops)} cropped objects"
        if len(self.crops) > 0:
            details = ", ".join([f"{crop.label} ({crop.score:.2f})" for crop in self.crops[:5]])
            if len(self.crops) > 5:
                details += ", ..."
            summary += f" [{details}]"
        return summary


class PreviewResult:
    """
    Tập hợp kết quả xem trước trực quan hóa (Preview) từ thư mục ảnh, file video hoặc danh sách ảnh media.load.
    """

    def __init__(
        self,
        results: List[DetectionResult],
        source_type: str = "image",  # "image", "video", "directory", "list"
        output_path: Optional[str] = None,
        annotated_frames: Optional[List[PIL.Image.Image]] = None,
        fps: float = 30.0,
    ) -> None:
        self.results = results
        self.source_type = source_type
        self.output_path = output_path
        self.annotated_frames = annotated_frames or []
        self.fps = fps

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> DetectionResult:
        return self.results[index]

    @property
    def images(self) -> List[PIL.Image.Image]:
        """Danh sách các ảnh / khung hình đã được vẽ bounding box."""
        if not self.annotated_frames and self.results:
            self.annotated_frames = [r.plot() for r in self.results]
        return self.annotated_frames

    def show(self, width: Optional[int] = None, limit: Optional[int] = None) -> None:
        """
        Hiển thị trực quan hóa kết quả nhận diện.
        - Trên Google Colab / Jupyter: Render ảnh inline trong cell.
        - Trên Desktop: Mở ảnh / video.
        """
        display_items = self.results if limit is None else self.results[:limit]
        for res in display_items:
            res.show(width=width)

    def save(self, output_path: str, fps: Optional[float] = None) -> str:
        """
        Lưu toàn bộ kết quả trực quan hóa theo đúng định dạng đích (folder ảnh hoặc file video .mp4).
        """
        target_fps = fps if fps is not None else self.fps
        p_str = str(output_path).lower()
        is_video_target = p_str.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"))

        if self.source_type == "video" or is_video_target:
            # Lưu video
            if not is_video_target:
                files.mkdir(output_path)
                final_path = os.path.join(output_path, "preview_video.mp4")
            else:
                parent_dir = os.path.dirname(os.path.abspath(output_path))
                if parent_dir:
                    files.mkdir(parent_dir)
                final_path = output_path
            frames = self.images
            media.save_video(final_path, frames, fps=target_fps, overwrite=True, verbose=False)
            self.output_path = final_path
            return final_path
        else:
            # Lưu folder ảnh
            files.mkdir(output_path)
            for idx, res in enumerate(self.results, 1):
                annotated = res.plot()
                img_path = os.path.join(output_path, f"annotated_{idx:05d}.jpg")
                media.save(img_path, annotated, overwrite=True, verbose=False)
            self.output_path = output_path
            return output_path

    def __repr__(self) -> str:
        out_info = f", output='{self.output_path}'" if self.output_path else ""
        return f"<PreviewResult count={len(self.results)}, source_type='{self.source_type}'{out_info}>"

