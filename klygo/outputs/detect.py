"""
Các lớp kết quả đầu ra chuẩn hóa cho nhận diện đối tượng (`klygo.outputs.detect`).

Hệ thống phân cấp 4 tầng:
- Box        → 1 BBox / 1 vật thể
- Crops      → Tập hợp các ảnh con đã cắt (alias: Boxes)
- Detection  → Kết quả nhận diện trên 1 ảnh / 1 frame
- Detections → Kết quả nhận diện toàn bộ video / folder
"""

import os
import math
import json
import datetime
from typing import List, Dict, Any, Optional, Union, Callable
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


# =============================================================================
# 1. CẤP HẠT NHÂN: Box - Đại diện cho 1 BBox / 1 Vật thể
# =============================================================================
class Box:
    """
    Đại diện cho một vật thể / một Bounding Box / một ảnh con đã cắt.
    detections[0][0] == detection[0] == crops[0] == box
    """

    def __init__(
        self,
        id: int,
        label: str,
        score: float,
        box: List[float],
        parent_image: Optional[PIL.Image.Image] = None,
        pad: int = 0,
    ) -> None:
        self.id = id
        self.label = str(label)
        self.score = float(score)
        self.box = [float(x) for x in box]  # [xmin, ymin, xmax, ymax]
        self.parent_image = parent_image
        self.pad = int(pad)

    @property
    def image(self) -> Optional[PIL.Image.Image]:
        """Lazy Cropping: Chỉ thực sự cắt ảnh khi người dùng gọi thuộc tính này."""
        if self.parent_image is None:
            return None
        w, h = self.parent_image.size
        x1 = max(0, int(self.box[0] - self.pad))
        y1 = max(0, int(self.box[1] - self.pad))
        x2 = min(w, int(self.box[2] + self.pad))
        y2 = min(h, int(self.box[3] + self.pad))
        if x2 <= x1 or y2 <= y1:
            return None
        return self.parent_image.crop((x1, y1, x2, y2))

    @property
    def width(self) -> float:
        """Chiều rộng pixel của BBox."""
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        """Chiều cao pixel của BBox."""
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """Diện tích pixel (width * height)."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Tỉ lệ khung hình (width / height)."""
        return round(self.width / max(1.0, self.height), 4)

    @property
    def xmin(self) -> float:
        return self.box[0]

    @property
    def ymin(self) -> float:
        return self.box[1]

    @property
    def xmax(self) -> float:
        return self.box[2]

    @property
    def ymax(self) -> float:
        return self.box[3]

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def center(self) -> tuple:
        return (self.center_x, self.center_y)

    @property
    def size(self) -> tuple:
        """Kích thước (width, height) của ảnh con."""
        img = self.image
        return img.size if img else (int(self.width), int(self.height))

    @property
    def confidence(self) -> float:
        """Alias của score."""
        return self.score

    def with_label(self, new_label: str) -> "Box":
        """Tạo bản sao mới với nhãn được đổi."""
        return Box(self.id, str(new_label), self.score, self.box, self.parent_image, self.pad)

    def with_score(self, new_score: float) -> "Box":
        """Tạo bản sao mới với điểm số được đổi."""
        return Box(self.id, self.label, float(new_score), self.box, self.parent_image, self.pad)

    def normalize(self, img_size: Optional[tuple] = None) -> List[float]:
        """Tọa độ chuẩn hóa tỉ lệ [0.0, 1.0] dạng [xmin, ymin, xmax, ymax]."""
        if img_size is None and self.parent_image is not None:
            img_size = self.parent_image.size
        if img_size and img_size[0] > 0 and img_size[1] > 0:
            w, h = img_size
            return [
                round(self.box[0] / w, 4),
                round(self.box[1] / h, 4),
                round(self.box[2] / w, 4),
                round(self.box[3] / h, 4),
            ]
        return []

    def show(self, width: Optional[int] = None) -> None:
        """Hiển thị riêng ảnh con này (Notebook hoặc Desktop)."""
        img = self.image
        if img is None:
            return
        if width is not None and width > 0 and img.width > 0:
            new_h = max(1, int(img.height * width / img.width))
            img = img.resize((width, new_h), PIL.Image.Resampling.BILINEAR)

        if _is_notebook():
            try:
                from IPython.display import display
                display(img)
            except Exception:
                img.show()
        else:
            img.show()

    def save(self, output_path: str) -> None:
        """Lưu ảnh con ra file đĩa."""
        img = self.image
        if img:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            media.save(output_path, img, overwrite=True, verbose=False)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin vật thể sang Dictionary tinh gọn."""
        return {
            "id": self.id,
            "label": self.label,
            "score": round(self.score, 4),
            "box": [round(x, 2) for x in self.box],
        }

    def __array__(self, dtype=None):
        """Hỗ trợ tự động cast sang NumPy Array cho Matplotlib plt.imshow(box)."""
        import numpy as np
        img = self.image
        if img is None:
            arr = np.zeros((10, 10, 3), dtype=np.uint8)
        else:
            arr = np.asarray(img)
        return arr.astype(dtype) if dtype is not None else arr

    def __getattr__(self, name: str) -> Any:
        """Chuyển tiếp các thuộc tính/hàm của PIL Image nếu có."""
        img = self.image
        if img is not None and hasattr(img, name):
            return getattr(img, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Box(id={self.id}, label='{self.label}', score={self.score:.3f}, box={[round(x, 1) for x in self.box]})"


# =============================================================================
# 2. CẤP TẬP HỢP ẢNH CON: Crops (alias: Boxes)
# =============================================================================
class Crops:
    """
    Tập hợp toàn bộ các ảnh con đã cắt (Cấp Classification / Patch List).
    """

    def __init__(
        self,
        crops: List[Box],
        source_image: Optional[PIL.Image.Image] = None,
        pad: int = 0,
    ) -> None:
        self.crops = crops
        self.source_image = source_image
        self.pad = pad

    def __getitem__(self, index: Union[int, slice]) -> Union[Box, "Crops"]:
        """crops[0] trả về 'Box', crops[m:n] trả về 'Crops' con."""
        if isinstance(index, slice):
            return Crops(self.crops[index], self.source_image, self.pad)
        return self.crops[index]

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        """Gán hoặc thay thế ảnh con: crops[0] = new_box hoặc crops[m:n] = [...]"""
        self.crops[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Xóa ảnh con: del crops[0] hoặc del crops[m:n]"""
        del self.crops[index]

    def append(self, crop: Box) -> None:
        """Thêm 1 ảnh con mới vào tập hợp."""
        self.crops.append(crop)

    def extend(self, other: Union[List[Box], "Crops"]) -> None:
        """Nối thêm nhiều ảnh con từ list hoặc từ Crops khác."""
        items = other.crops if isinstance(other, Crops) else list(other)
        self.crops.extend(items)
        for i, c in enumerate(self.crops):
            c.id = i

    def pop(self, index: int = -1) -> Box:
        """Lấy ra và xóa ảnh con tại vị trí index."""
        return self.crops.pop(index)

    def insert(self, index: int, crop: Box) -> None:
        """Chèn 1 ảnh con vào vị trí index."""
        self.crops.insert(index, crop)
        for i, c in enumerate(self.crops):
            c.id = i

    def remove(self, crop: Box) -> None:
        """Xóa 1 ảnh con cụ thể."""
        self.crops.remove(crop)
        for i, c in enumerate(self.crops):
            c.id = i

    def clear(self) -> None:
        """Xóa sạch toàn bộ ảnh con."""
        self.crops.clear()

    def __len__(self) -> int:
        return len(self.crops)

    def __add__(self, other: "Crops") -> "Crops":
        """Ghép 2 tập hợp ảnh con: crops = crops_1 + crops_2"""
        if not isinstance(other, Crops):
            raise TypeError(f"Không thể cộng Crops với kiểu {type(other)}")
        combined = [
            Box(
                id=i,
                label=c.label,
                score=c.score,
                box=list(c.box),
                parent_image=c.parent_image or self.source_image,
                pad=c.pad,
            )
            for i, c in enumerate(list(self.crops) + list(other.crops))
        ]
        return Crops(combined, self.source_image, self.pad)

    def __iter__(self):
        return iter(self.crops)

    @property
    def images(self) -> List[PIL.Image.Image]:
        """Danh sách toàn bộ các đối tượng ảnh con PIL.Image thuần túy."""
        return [c.image for c in self.crops if c.image is not None]

    @property
    def labels(self) -> List[str]:
        return [c.label for c in self.crops]

    @property
    def scores(self) -> List[float]:
        return [c.score for c in self.crops]

    @property
    def boxes(self) -> List[List[float]]:
        return [c.box for c in self.crops]

    @property
    def areas(self) -> List[float]:
        return [c.area for c in self.crops]

    @property
    def count(self) -> int:
        return len(self.crops)

    @property
    def unique_labels(self) -> List[str]:
        seen = set()
        out = []
        for c in self.crops:
            if c.label not in seen:
                seen.add(c.label)
                out.append(c.label)
        return out

    @property
    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.crops:
            counts[c.label] = counts.get(c.label, 0) + 1
        return counts

    def filter(self, fn: Callable[[Box], bool]) -> "Crops":
        """Lọc tập ảnh con bằng Lambda -> Trả về Crops mới."""
        filtered = [c for c in self.crops if fn(c)]
        reindexed = [
            Box(i, c.label, c.score, c.box, c.parent_image, c.pad)
            for i, c in enumerate(filtered)
        ]
        return Crops(reindexed, self.source_image, self.pad)

    def sort(
        self,
        key: Optional[Callable[[Box], Any]] = None,
        reverse: bool = True,
    ) -> "Crops":
        """Sắp xếp tập ảnh con bằng Lambda -> Trả về Crops mới."""
        sort_fn = key or (lambda c: c.score)
        sorted_crops = sorted(self.crops, key=sort_fn, reverse=reverse)
        reindexed = [
            Box(i, c.label, c.score, c.box, c.parent_image, c.pad)
            for i, c in enumerate(sorted_crops)
        ]
        return Crops(reindexed, self.source_image, self.pad)

    def group_by(
        self,
        key: Optional[Callable[[Box], Any]] = None,
    ) -> Dict[Any, "Crops"]:
        """Gom nhóm các ảnh con theo Class hoặc điều kiện."""
        key_fn = key or (lambda c: c.label)
        groups: Dict[Any, List[Box]] = {}
        for c in self.crops:
            k = key_fn(c)
            groups.setdefault(k, []).append(c)
        return {
            k: Crops(v, self.source_image, self.pad)
            for k, v in groups.items()
        }

    def map(
        self,
        fn: Union[Dict[str, str], Callable[[Box], Any]],
    ) -> "Crops":
        """Biến đổi dữ liệu ảnh con bằng Dict đổi nhãn hoặc Lambda."""
        mapped = []
        for i, c in enumerate(self.crops):
            if isinstance(fn, dict):
                new_label = fn.get(c.label, c.label)
                mapped.append(Box(i, new_label, c.score, c.box, c.parent_image, c.pad))
            elif callable(fn):
                res = fn(c)
                if isinstance(res, Box):
                    res.id = i
                    mapped.append(res)
                elif isinstance(res, str):
                    mapped.append(Box(i, res, c.score, c.box, c.parent_image, c.pad))
        return Crops(mapped, self.source_image, self.pad)

    def show(
        self,
        grid: bool = True,
        limit: Optional[int] = None,
        cell_size: int = 200,
    ) -> None:
        """
        Hiển thị tập ảnh con:
        - Mặc định (grid=True): Ghép thành lưới vuông sqrt(N) x sqrt(N) tự đệm trắng viền không cắt góc.
        - grid=False: Hiển thị từng ảnh con rời.
        """
        items = self.crops[:limit] if limit else self.crops
        valid_items = [c for c in items if c.image is not None]
        if not valid_items:
            return

        if not grid or len(valid_items) == 1:
            for c in valid_items:
                c.show(width=cell_size)
            return

        n = len(valid_items)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        grid_w, grid_h = cols * cell_size, rows * cell_size

        # Tạo Canvas nền trắng tinh phẳng đẹp
        canvas = PIL.Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        for idx, crop_item in enumerate(valid_items):
            img = crop_item.image
            if img is None:
                continue
            r, c = idx // cols, idx % cols
            # Letterbox giữ nguyên tỉ lệ gốc của ảnh con
            img_ratio = img.width / max(1, img.height)
            if img_ratio > 1:
                nw, nh = cell_size - 12, int((cell_size - 12) / img_ratio)
            else:
                nh, nw = cell_size - 12, int((cell_size - 12) * img_ratio)
            resized = img.resize((max(1, nw), max(1, nh)), PIL.Image.Resampling.BILINEAR)

            x_pos = c * cell_size + (cell_size - nw) // 2
            y_pos = r * cell_size + (cell_size - nh) // 2
            canvas.paste(resized, (x_pos, y_pos))

        if _is_notebook():
            try:
                from IPython.display import display
                display(canvas)
            except Exception:
                canvas.show()
        else:
            canvas.show()

    def export(
        self,
        output_path: str,
        format: str = "classification",
    ) -> None:
        """
        Tự động xuất toàn bộ ảnh con thành Bộ Dữ Liệu Phân Loại Ảnh (Classification Dataset),
        tự tạo thư mục con theo nhãn class: output_path/orange/orange_00000.jpg.
        """
        os.makedirs(output_path, exist_ok=True)
        class_counters: Dict[str, int] = {}
        for crop in self.crops:
            if crop.image:
                label_clean = str(crop.label).strip().replace(" ", "_") or "unlabeled"
                class_dir = os.path.join(output_path, label_clean)
                os.makedirs(class_dir, exist_ok=True)
                idx = class_counters.get(label_clean, 0)
                file_path = os.path.join(class_dir, f"{label_clean}_{idx:05d}.jpg")
                media.save(file_path, crop.image, overwrite=True, verbose=False)
                class_counters[label_clean] = idx + 1

    def save(self, output_dir: str, by_class: bool = True) -> List[str]:
        """Lưu toàn bộ ảnh con ra thư mục."""
        self.export(output_dir, format="classification" if by_class else "flat")
        return [os.path.join(output_dir, str(c.label)) for c in self.crops]

    def to_dict(self) -> Dict[str, Any]:
        """Xuất thông tin tập ảnh con sang JSON Key-Value."""
        return {
            "total_crops": len(self.crops),
            "pad": self.pad,
            "unique_labels": self.unique_labels,
            "label_counts": self.label_counts,
            "crops": [c.to_dict() for c in self.crops],
        }

    def __repr__(self) -> str:
        summary = f"Crops: {len(self.crops)} crops"
        if len(self.crops) > 0:
            details = ", ".join([f"{c.label} ({c.score:.2f})" for c in self.crops[:5]])
            if len(self.crops) > 5:
                details += ", ..."
            summary += f" [{details}]"
        return summary


# =============================================================================
# 3. CẤP 1 BỨC ẢNH / 1 FRAME: Detection
# =============================================================================
class Detection:
    """
    Đầu ra chuẩn hóa chứa tất cả kết quả nhận diện của một bức ảnh / 1 frame.
    detections[0] == detection
    detection[0] == box
    """

    def __init__(
        self,
        source_image: PIL.Image.Image,
        objects: List[Box],
        speed: Optional[dict] = None,
        image_frame_index: int = 0,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.source_image = source_image
        self.objects = objects  # Danh sách các Box
        self.speed = speed or {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0, "total": 0.0}
        self.image_frame_index = image_frame_index
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Cấu hình tự do linh hoạt (Open Configuration)
        self.config: Dict[str, Any] = dict(config or {})
        self.config.update(kwargs)

        # Gán liên kết ảnh mẹ vào từng box
        for c in self.objects:
            c.parent_image = self.source_image

    @property
    def text_prompt(self) -> Optional[Union[str, List[str]]]:
        """Từ khóa/Nhãn nhận diện (truy xuất từ config)."""
        return self.config.get("text_prompt", self.config.get("prompt"))

    @property
    def threshold(self) -> Optional[float]:
        """Ngưỡng độ tin cậy (truy xuất từ config)."""
        return self.config.get("threshold", self.config.get("conf"))

    @property
    def conf(self) -> Optional[float]:
        """Alias của threshold."""
        return self.threshold

    @property
    def text_threshold(self) -> Optional[float]:
        """Ngưỡng tương đồng văn bản Open-Vocabulary (truy xuất từ config)."""
        return self.config.get("text_threshold")

    def __getitem__(self, index: Union[int, slice]) -> Union[Box, "Detection"]:
        """🎯 detection[0] trả về 'Box', detection[m:n] trả về 'Detection' con."""
        if isinstance(index, slice):
            return Detection(
                self.source_image,
                self.objects[index],
                self.speed,
                self.image_frame_index,
                config=self.config,
            )
        return self.objects[index]

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        """Gán hoặc thay thế box: detection[0] = new_box hoặc detection[m:n] = [...]"""
        if isinstance(value, Box):
            value.parent_image = self.source_image
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Box):
                    item.parent_image = self.source_image
        self.objects[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Xóa box: del detection[0] hoặc del detection[m:n]"""
        del self.objects[index]

    def append(self, crop: Box) -> None:
        """Thêm 1 box mới vào detection."""
        if isinstance(crop, Box):
            crop.parent_image = self.source_image
        self.objects.append(crop)

    def extend(self, other: Union[List[Box], "Detection", Crops]) -> None:
        """Nối thêm nhiều box từ list hoặc từ detection khác."""
        if isinstance(other, Detection):
            items = other.objects
        elif isinstance(other, Crops):
            items = other.crops
        else:
            items = list(other)
        for c in items:
            if isinstance(c, Box):
                c.parent_image = self.source_image
            self.objects.append(c)
        for i, c in enumerate(self.objects):
            c.id = i

    def pop(self, index: int = -1) -> Box:
        """Lấy ra và xóa box tại vị trí index."""
        return self.objects.pop(index)

    def insert(self, index: int, crop: Box) -> None:
        """Chèn 1 box vào vị trí index."""
        if isinstance(crop, Box):
            crop.parent_image = self.source_image
        self.objects.insert(index, crop)
        for i, c in enumerate(self.objects):
            c.id = i

    def remove(self, crop: Box) -> None:
        """Xóa 1 box cụ thể khỏi detection."""
        self.objects.remove(crop)
        for i, c in enumerate(self.objects):
            c.id = i

    def clear(self) -> None:
        """Xóa sạch toàn bộ box trong detection."""
        self.objects.clear()

    def __len__(self) -> int:
        return len(self.objects)

    def __add__(self, other: "Detection") -> "Detection":
        """Ghép 2 kết quả nhận diện trên cùng 1 ảnh: det = det_1 + det_2"""
        if not isinstance(other, Detection):
            raise TypeError(f"Không thể cộng Detection với kiểu {type(other)}")
        combined = [
            Box(
                id=i,
                label=c.label,
                score=c.score,
                box=list(c.box),
                parent_image=self.source_image,
                pad=c.pad,
            )
            for i, c in enumerate(list(self.objects) + list(other.objects))
        ]
        return Detection(
            self.source_image,
            combined,
            self.speed,
            self.image_frame_index,
            config=self.config,
        )

    def __iter__(self):
        return iter(self.objects)

    @property
    def boxes(self) -> List[List[float]]:
        """Trả về danh sách tọa độ pixel tuyệt đối [xmin, ymin, xmax, ymax]."""
        return [c.box for c in self.objects]

    @property
    def normalized_boxes(self) -> List[List[float]]:
        """Trả về danh sách tọa độ chuẩn hóa tỉ lệ trong khoảng [0.0, 1.0]."""
        w, h = self.source_image.size
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
        return [c.label for c in self.objects]

    @property
    def scores(self) -> List[float]:
        return [c.score for c in self.objects]

    @property
    def areas(self) -> List[float]:
        return [c.area for c in self.objects]

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def image_size(self) -> tuple:
        return self.source_image.size

    @property
    def image_format(self) -> str:
        return getattr(self.source_image, "format", "JPEG") or "JPEG"

    @property
    def image_mode(self) -> str:
        return getattr(self.source_image, "mode", "RGB")

    @property
    def unique_labels(self) -> List[str]:
        seen = set()
        out = []
        for c in self.objects:
            if c.label not in seen:
                seen.add(c.label)
                out.append(c.label)
        return out

    @property
    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.objects:
            counts[c.label] = counts.get(c.label, 0) + 1
        return counts

    def crop(self, pad: int = 0) -> Crops:
        """Cắt toàn bộ các box trong ảnh -> Trả về Crops."""
        crop_items = [
            Box(c.id, c.label, c.score, c.box, self.source_image, pad=pad)
            for c in self.objects
        ]
        return Crops(crop_items, self.source_image, pad=pad)

    def filter(self, fn: Callable[[Box], bool]) -> "Detection":
        """Lọc các box trong ảnh bằng Lambda -> Trả về Detection mới đã lọc."""
        filtered = [c for c in self.objects if fn(c)]
        reindexed = [
            Box(i, c.label, c.score, c.box, self.source_image, c.pad)
            for i, c in enumerate(filtered)
        ]
        return Detection(
            self.source_image,
            reindexed,
            self.speed,
            self.image_frame_index,
            self.text_prompt,
            self.threshold,
            self.text_threshold,
        )

    def sort(
        self,
        key: Optional[Callable[[Box], Any]] = None,
        reverse: bool = True,
    ) -> "Detection":
        """Sắp xếp các box trong ảnh bằng Lambda -> Trả về Detection mới."""
        sort_fn = key or (lambda c: c.score)
        sorted_objs = sorted(self.objects, key=sort_fn, reverse=reverse)
        reindexed = [
            Box(i, c.label, c.score, c.box, self.source_image, c.pad)
            for i, c in enumerate(sorted_objs)
        ]
        return Detection(
            self.source_image,
            reindexed,
            self.speed,
            self.image_frame_index,
            self.text_prompt,
            self.threshold,
            self.text_threshold,
        )

    def group_by(
        self,
        key: Optional[Callable[[Box], Any]] = None,
    ) -> Dict[Any, "Detection"]:
        """Gom nhóm các box theo Class hoặc điều kiện."""
        key_fn = key or (lambda c: c.label)
        groups: Dict[Any, List[Box]] = {}
        for c in self.objects:
            k = key_fn(c)
            groups.setdefault(k, []).append(c)
        return {
            k: Detection(self.source_image, v, self.speed, self.image_frame_index)
            for k, v in groups.items()
        }

    def map(
        self,
        fn: Union[Dict[str, str], Callable[[Box], Any]],
    ) -> "Detection":
        """Biến đổi dữ liệu box bằng Dict đổi nhãn hoặc Lambda."""
        mapped = []
        for i, c in enumerate(self.objects):
            if isinstance(fn, dict):
                new_label = fn.get(c.label, c.label)
                mapped.append(Box(i, new_label, c.score, c.box, self.source_image, c.pad))
            elif callable(fn):
                res = fn(c)
                if isinstance(res, Box):
                    res.id = i
                    mapped.append(res)
                elif isinstance(res, str):
                    mapped.append(Box(i, res, c.score, c.box, self.source_image, c.pad))
        return Detection(
            self.source_image,
            mapped,
            self.speed,
            self.image_frame_index,
            self.text_prompt,
            self.threshold,
            self.text_threshold,
        )

    def export(self, output_path: str, format: str = "yolo") -> None:
        """Xuất file nhãn YOLO (.txt) hoặc JSON cho bức ảnh này."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if format.lower() == "yolo":
            w, h = self.source_image.size
            unique_classes = sorted(list(set(self.labels)))
            lines = []
            for c in self.objects:
                cid = unique_classes.index(c.label)
                cx = (c.box[0] + c.box[2]) / 2.0 / w
                cy = (c.box[1] + c.box[3]) / 2.0 / h
                bw = (c.box[2] - c.box[0]) / w
                bh = (c.box[3] - c.box[1]) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        elif format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("id,label,score,xmin,ymin,xmax,ymax,area\n")
                for c in self.objects:
                    f.write(f"{c.id},{c.label},{c.score:.4f},{c.box[0]:.2f},{c.box[1]:.2f},{c.box[2]:.2f},{c.box[3]:.2f},{c.area:.2f}\n")

    def to_dict(self) -> Dict[str, Any]:
        """Xuất toàn bộ kết quả về dạng dict chuẩn COCO/Roboflow."""
        d = {
            "image_format": self.image_format,
            "image_mode": self.image_mode,
            "image_size": list(self.source_image.size),
            "image_frame_index": self.image_frame_index,
            "timestamp": self.timestamp,
            "objects": [c.to_dict() for c in self.objects],
            "speed": self.speed,
        }
        if self.config:
            d["config"] = self.config
        return d

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
        """Vẽ bounding box và nhãn tên theo chuẩn đồ họa Ultralytics YOLO (2-Pass Rendering)."""
        show_scores = scores
        if score is not None:
            show_scores = score
        elif conf is not None:
            show_scores = conf

        annotated = self.source_image.copy()
        draw = PIL.ImageDraw.Draw(annotated)
        w_img, h_img = self.source_image.size

        # 1. Tính toán độ dày viền & cỡ chữ tự động theo tỉ lệ kích thước ảnh
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

        # 5. PASS 2: Vẽ toàn bộ Badge nhãn chữ lên trên cùng
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
        """Vẽ và hiển thị ảnh nhận diện (tương thích cả Notebook lẫn Desktop)."""
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

    def __repr__(self) -> str:
        summary = f"Detection: {len(self.objects)} objects detected"
        if len(self.objects) > 0:
            details = ", ".join([f"{c.label} ({c.score:.2f})" for c in self.objects[:5]])
            if len(self.objects) > 5:
                details += ", ..."
            summary += f" [{details}]"
        return summary


# =============================================================================
# 4. CẤP TOÀN BỘ VIDEO / FOLDER: Detections
# =============================================================================
class Detections:
    """
    Tập hợp tất cả các Detection của toàn bộ Video / Folder ảnh.
    detections[0] == detection
    detections[0][0] == box
    """

    def __init__(
        self,
        frames: List[Detection],
        source_type: str = "video",
        fps: float = 30.0,
        output_path: Optional[str] = None,
    ) -> None:
        self.frames = frames
        self.source_type = source_type
        self.fps = fps
        self.output_path = output_path
        self.results = frames  # alias

    def __getitem__(self, index: Union[int, slice]) -> Union[Detection, "Detections"]:
        """🎯 detections[0] trả về 'Detection', detections[m:n] trả về 'Detections' con."""
        if isinstance(index, slice):
            return Detections(self.frames[index], self.source_type, self.fps, self.output_path)
        return self.frames[index]

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        """Gán hoặc thay thế frame: detections[0] = new_detection hoặc detections[m:n] = [...]"""
        self.frames[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Xóa frame: del detections[0] hoặc del detections[m:n]"""
        del self.frames[index]

    def append(self, frame: Detection) -> None:
        """Thêm 1 detection mới vào cuối detections."""
        self.frames.append(frame)

    def extend(self, other: Union[List[Detection], "Detections"]) -> None:
        """Nối thêm nhiều detection vào detections."""
        items = other.frames if isinstance(other, Detections) else list(other)
        self.frames.extend(items)

    def pop(self, index: int = -1) -> Detection:
        """Lấy ra và xóa frame tại index (mặc định frame cuối)."""
        return self.frames.pop(index)

    def insert(self, index: int, frame: Detection) -> None:
        """Chèn 1 detection vào vị trí index."""
        self.frames.insert(index, frame)

    def __len__(self) -> int:
        return len(self.frames)

    def __add__(self, other: "Detections") -> "Detections":
        """Ghép 2 tập hợp video / folder: detections = detections_1 + detections_2"""
        if not isinstance(other, Detections):
            raise TypeError(f"Không thể cộng Detections với kiểu {type(other)}")
        combined = list(self.frames) + list(other.frames)
        return Detections(combined, self.source_type, self.fps, self.output_path)

    def __iter__(self):
        return iter(self.frames)

    @property
    def total_objects(self) -> int:
        """Tổng số vật thể tìm thấy trên toàn bộ video/folder."""
        return sum(len(f) for f in self.frames)

    @property
    def unique_labels(self) -> List[str]:
        """Tự động quét danh sách class duy nhất theo thời gian thực từ các frame con."""
        seen = set()
        out = []
        for f in self.frames:
            for l in f.labels:
                if l not in seen:
                    seen.add(l)
                    out.append(l)
        return out

    @property
    def label_counts(self) -> Dict[str, int]:
        """Tự động đếm tổng số lượng từng class trên toàn bộ video."""
        counts: Dict[str, int] = {}
        for f in self.frames:
            for l in f.labels:
                counts[l] = counts.get(l, 0) + 1
        return counts

    @property
    def labels(self) -> List[List[str]]:
        """Danh sách nhãn của từng frame."""
        return [f.labels for f in self.frames]

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def images(self) -> List[PIL.Image.Image]:
        """Danh sách các ảnh / khung hình đã được vẽ bounding box."""
        return [f.plot() for f in self.frames]

    def filter(
        self,
        fn: Callable[[Any], bool],
    ) -> "Detections":
        """
        Lọc dữ liệu trên toàn bộ video/folder:
        - Nếu fn nhận frame (Detection): lọc chọn các frame thỏa mãn.
        - Nếu fn nhận box (Box): tự động lọc các box trong từng frame.
        """
        # Thử nghiệm xem lambda nhận frame hay box
        filtered_frames = []
        for f in self.frames:
            # Thử gọi trên frame trước
            try:
                if fn(f):
                    filtered_frames.append(f)
            except Exception:
                # Nếu lỗi thuộc tính của frame, áp dụng lọc trên từng box của frame
                cleaned_frame = f.filter(fn)
                if len(cleaned_frame) > 0:
                    filtered_frames.append(cleaned_frame)

        return Detections(filtered_frames, self.source_type, self.fps, self.output_path)

    def sort(
        self,
        key: Optional[Callable[[Detection], Any]] = None,
        reverse: bool = True,
    ) -> "Detections":
        """Sắp xếp các frame trong video."""
        sort_fn = key or (lambda f: len(f))
        sorted_frames = sorted(self.frames, key=sort_fn, reverse=reverse)
        return Detections(sorted_frames, self.source_type, self.fps, self.output_path)

    def group_by(
        self,
        key: Optional[Callable[[Detection], Any]] = None,
    ) -> Dict[Any, "Detections"]:
        """Gom nhóm các frame theo điều kiện."""
        key_fn = key or (lambda f: f.image_frame_index)
        groups: Dict[Any, List[Detection]] = {}
        for f in self.frames:
            k = key_fn(f)
            groups.setdefault(k, []).append(f)
        return {
            k: Detections(v, self.source_type, self.fps)
            for k, v in groups.items()
        }

    def map(
        self,
        fn: Union[Dict[str, str], Callable[[Detection], Detection]],
    ) -> "Detections":
        """Biến đổi hàng loạt trên các frame."""
        mapped = []
        for f in self.frames:
            mapped.append(f.map(fn))
        return Detections(mapped, self.source_type, self.fps, self.output_path)

    def crop(
        self,
        output_path: Optional[str] = None,
        pad: int = 0,
    ) -> Crops:
        """Cắt toàn bộ vật thể trên tất cả các frame -> Trả về Crops."""
        all_crops: List[Box] = []
        for f in self.frames:
            crops_obj = f.crop(pad=pad)
            all_crops.extend(crops_obj.crops)

        # Đánh lại ID liên tục
        for i, c in enumerate(all_crops):
            c.id = i

        crops_collection = Crops(all_crops, pad=pad)
        if output_path:
            crops_collection.export(output_path, format="classification")
        return crops_collection

    def export(
        self,
        output_path: str,
        format: str = "yolo",
    ) -> None:
        """
        Xuất toàn bộ Video thành Flat YOLO Dataset (images/, labels/, data.yaml) hoặc JSON.
        """
        if format.lower() == "yolo":
            img_dir = os.path.join(output_path, "images")
            lbl_dir = os.path.join(output_path, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            classes = self.unique_labels
            for idx, res in enumerate(self.frames):
                img_name = f"frame_{idx:05d}.jpg"
                lbl_name = f"frame_{idx:05d}.txt"

                # Lưu ảnh frame
                res.source_image.save(os.path.join(img_dir, img_name))

                # Lưu file nhãn YOLO
                w, h = res.source_image.size
                lines = []
                for c in res.objects:
                    cid = classes.index(c.label)
                    cx = (c.box[0] + c.box[2]) / 2.0 / w
                    cy = (c.box[1] + c.box[3]) / 2.0 / h
                    bw = (c.box[2] - c.box[0]) / w
                    bh = (c.box[3] - c.box[1]) / h
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                with open(os.path.join(lbl_dir, lbl_name), "w", encoding="utf-8") as f:
                    f.writelines(lines)

            # Tự động tạo file data.yaml chuẩn Ultralytics
            yaml_content = f"path: {os.path.abspath(output_path)}\ntrain: images\nval: images\n\nnames:\n"
            for i, name in enumerate(classes):
                yaml_content += f"  {i}: {name}\n"

            with open(os.path.join(output_path, "data.yaml"), "w", encoding="utf-8") as f:
                f.write(yaml_content)

        elif format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save(
        self,
        output_path: str,
        fps: Optional[float] = None,
    ) -> str:
        """
        Lưu toàn bộ video hoặc thư mục ảnh thành phẩm đã vẽ Bounding Box.
        """
        target_fps = fps if fps is not None else self.fps
        p_str = str(output_path).lower()
        is_video_target = p_str.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"))

        if self.source_type == "video" or is_video_target:
            if not is_video_target:
                files.mkdir(output_path)
                final_path = os.path.join(output_path, "annotated_video.mp4")
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
            files.mkdir(output_path)
            for idx, res in enumerate(self.frames, 1):
                annotated = res.plot()
                img_path = os.path.join(output_path, f"annotated_{idx:05d}.jpg")
                media.save(img_path, annotated, overwrite=True, verbose=False)
            self.output_path = output_path
            return output_path

    def show(self, width: Optional[int] = None, limit: Optional[int] = 5) -> None:
        """Xem trước các frame đầu tiên."""
        display_items = self.frames if limit is None else self.frames[:limit]
        for res in display_items:
            res.show(width=width)

    def to_dict(self) -> Dict[str, Any]:
        """Xuất cấu trúc Key-Value chuẩn toàn bộ video."""
        return {
            "source_type": self.source_type,
            "total_frames": len(self.frames),
            "fps": self.fps,
            "total_objects": self.total_objects,
            "unique_labels": self.unique_labels,
            "label_counts": self.label_counts,
            "frames": [f.to_dict() for f in self.frames],
        }

    def __repr__(self) -> str:
        return f"<Detections frames={len(self.frames)}, total_objects={self.total_objects}, unique_labels={self.unique_labels}>"


# =============================================================================
# ALIASES TƯƠNG THÍCH NGƯỢC
# =============================================================================
CropResult = Box
DetectedObject = Box
CroppedObject = Box
Boxes = Crops
CropResults = Crops
DetectionResult = Detection
DetectionResults = Detections
PreviewResult = Detections
