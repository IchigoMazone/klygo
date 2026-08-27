"""
Các lớp kết quả đầu ra chuẩn hóa cho nhận diện đối tượng (`klygo.outputs.detect`).

Hệ thống phân cấp 4 tầng:
- Box        : 1 BBox / 1 vật thể
- Crops      : Tập hợp các ảnh con đã cắt (alias: Boxes)
- Detection  : Kết quả nhận diện trên 1 ảnh / 1 frame
- Detections : Kết quả nhận diện toàn bộ video / folder
"""

import os
import datetime
from typing import List, Dict, Any, Optional, Union, Callable
import PIL.Image

from klygo import files, media, visual


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
        self.id = int(id)
        self.label = str(label)
        self.score = float(score)
        self.box = [float(x) for x in box]  # [xmin, ymin, xmax, ymax]
        self.parent_image = parent_image
        self.pad = int(pad)

    @property
    def image(self) -> Optional[PIL.Image.Image]:
        """Lazy Cropping: Cắt ảnh con chính xác từ ảnh mẹ theo tọa độ pixel."""
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

    def with_pad(self, pad: int) -> "Box":
        """Tạo bản sao mới với độ dày đệm viền pad mới."""
        return Box(self.id, self.label, self.score, self.box, self.parent_image, int(pad))

    def crop(self, pad: Optional[int] = None) -> Optional[PIL.Image.Image]:
        """Cắt ảnh con với đệm viền tùy chọn."""
        if pad is not None and pad != self.pad:
            return self.with_pad(pad).image
        return self.image

    def show(self, width: Optional[int] = None) -> None:
        """Hiển thị ảnh con trên màn hình hoặc notebook."""
        img = self.image
        if img:
            visual.show_image(img, width=width)

    def save(self, output_path: str) -> None:
        """Lưu ảnh con ra đĩa thông qua klygo.media.save."""
        img = self.image
        if img:
            media.save(output_path, img, overwrite=True, verbose=False)

    def to_dict(self) -> Dict[str, Any]:
        """Xuất thông tin Box sang JSON Key-Value."""
        return {
            "id": self.id,
            "label": self.label,
            "score": round(self.score, 4),
            "box": [round(x, 2) for x in self.box],
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "area": round(self.area, 2),
        }

    def __repr__(self) -> str:
        coords = [round(x, 1) for x in self.box]
        return f"Box(id={self.id}, label='{self.label}', score={self.score:.2f}, box={coords})"


# =============================================================================
# 2. CẤP TẬP HỢP ẢNH CON: Crops
# =============================================================================
class Crops:
    """
    Tập hợp tất cả các ảnh con (Crops) đã cắt từ 1 hoặc nhiều ảnh.
    crops[0] == box
    """

    def __init__(
        self,
        crops: List[Box],
        source_image: Optional[PIL.Image.Image] = None,
        pad: int = 0,
    ) -> None:
        self.crops = crops
        self.source_image = source_image
        self.pad = int(pad)

    def __getitem__(self, index: Union[int, slice]) -> Union[Box, "Crops"]:
        if isinstance(index, slice):
            return Crops(self.crops[index], self.source_image, self.pad)
        return self.crops[index]

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        self.crops[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.crops[index]

    def append(self, crop: Box) -> None:
        self.crops.append(crop)

    def extend(self, other: Union[List[Box], "Crops"]) -> None:
        items = other.crops if isinstance(other, Crops) else list(other)
        self.crops.extend(items)

    def pop(self, index: int = -1) -> Box:
        return self.crops.pop(index)

    def insert(self, index: int, crop: Box) -> None:
        self.crops.insert(index, crop)

    def remove(self, crop: Box) -> None:
        self.crops.remove(crop)

    def clear(self) -> None:
        self.crops.clear()

    def __len__(self) -> int:
        return len(self.crops)

    def __add__(self, other: "Crops") -> "Crops":
        if not isinstance(other, Crops):
            raise TypeError(f"Không thể cộng Crops với kiểu {type(other)}")
        return Crops(list(self.crops) + list(other.crops), self.source_image, self.pad)

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
        return Crops(filtered, self.source_image, self.pad)

    def sort(
        self,
        key: Optional[Callable[[Box], Any]] = None,
        reverse: bool = True,
    ) -> "Crops":
        """Sắp xếp tập ảnh con bằng Lambda -> Trả về Crops mới."""
        sort_fn = key or (lambda c: c.score)
        return Crops(sorted(self.crops, key=sort_fn, reverse=reverse), self.source_image, self.pad)

    def group_by(
        self,
        key: Optional[Callable[[Box], Any]] = None,
    ) -> Dict[Any, "Crops"]:
        """Gom nhóm các ảnh con theo Class hoặc điều kiện."""
        key_fn = key or (lambda c: c.label)
        groups: Dict[Any, List[Box]] = {}
        for c in self.crops:
            groups.setdefault(key_fn(c), []).append(c)
        return {k: Crops(v, self.source_image, self.pad) for k, v in groups.items()}

    def map(
        self,
        fn: Union[Dict[str, str], Callable[[Box], Any]],
    ) -> "Crops":
        """Biến đổi dữ liệu ảnh con bằng Dict đổi nhãn hoặc Lambda."""
        mapped = []
        for i, c in enumerate(self.crops):
            if isinstance(fn, dict):
                mapped.append(Box(i, fn.get(c.label, c.label), c.score, c.box, c.parent_image, c.pad))
            elif callable(fn):
                res = fn(c)
                if isinstance(res, Box):
                    mapped.append(res)
                elif isinstance(res, str):
                    mapped.append(Box(i, res, c.score, c.box, c.parent_image, c.pad))
        return Crops(mapped, self.source_image, self.pad)

    def show(self, limit: Optional[int] = None, cell_size: int = 200) -> None:
        """Hiển thị tập ảnh con trên màn hình hoặc notebook."""
        items = self.images[:limit] if limit else self.images
        for img in items:
            visual.show_image(img, width=cell_size)

    def export(
        self,
        output_path: str,
        format: str = "classification",
    ) -> None:
        """
        Tự động xuất toàn bộ ảnh con thành Bộ Dữ Liệu Phân Loại Ảnh (Classification Dataset)
        dùng klygo.files và klygo.media.
        """
        files.mkdir(output_path)
        class_counters: Dict[str, int] = {}
        for crop in self.crops:
            if crop.image:
                label_clean = str(crop.label).strip().replace(" ", "_") or "unlabeled"
                class_dir = os.path.join(output_path, label_clean)
                files.mkdir(class_dir)
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

        # Cấu hình động tự do (Open Configuration)
        self.config: Dict[str, Any] = dict(config or {})
        self.config.update(kwargs)

        # Gán liên kết ảnh mẹ vào từng box
        for c in self.objects:
            c.parent_image = self.source_image

    @property
    def text_prompt(self) -> Optional[Union[str, List[str]]]:
        return self.config.get("text_prompt", self.config.get("prompt"))

    @property
    def threshold(self) -> Optional[float]:
        return self.config.get("threshold", self.config.get("conf"))

    @property
    def conf(self) -> Optional[float]:
        return self.threshold

    @property
    def text_threshold(self) -> Optional[float]:
        return self.config.get("text_threshold")

    def __getattr__(self, name: str) -> Any:
        cfg = object.__getattribute__(self, "__dict__").get("config", {})
        if name in cfg:
            return cfg[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        if isinstance(value, Box):
            value.parent_image = self.source_image
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Box):
                    item.parent_image = self.source_image
        self.objects[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.objects[index]

    def append(self, crop: Box) -> None:
        if isinstance(crop, Box):
            crop.parent_image = self.source_image
        self.objects.append(crop)

    def extend(self, other: Union[List[Box], "Detection", Crops]) -> None:
        items = other.objects if isinstance(other, Detection) else (other.crops if isinstance(other, Crops) else list(other))
        for c in items:
            if isinstance(c, Box):
                c.parent_image = self.source_image
            self.objects.append(c)
        for i, c in enumerate(self.objects):
            c.id = i

    def pop(self, index: int = -1) -> Box:
        return self.objects.pop(index)

    def insert(self, index: int, crop: Box) -> None:
        if isinstance(crop, Box):
            crop.parent_image = self.source_image
        self.objects.insert(index, crop)

    def remove(self, crop: Box) -> None:
        self.objects.remove(crop)

    def clear(self) -> None:
        self.objects.clear()

    def __len__(self) -> int:
        return len(self.objects)

    def __add__(self, other: "Detection") -> "Detection":
        if not isinstance(other, Detection):
            raise TypeError(f"Không thể cộng Detection với kiểu {type(other)}")
        return Detection(
            self.source_image,
            list(self.objects) + list(other.objects),
            self.speed,
            self.image_frame_index,
            config=self.config,
        )

    def __iter__(self):
        return iter(self.objects)

    @property
    def boxes(self) -> List[List[float]]:
        """Danh sách tọa độ pixel tuyệt đối [xmin, ymin, xmax, ymax]."""
        return [c.box for c in self.objects]

    @property
    def normalized_boxes(self) -> List[List[float]]:
        """Danh sách tọa độ chuẩn hóa tỉ lệ trong khoảng [0.0, 1.0]."""
        w, h = self.source_image.size
        if w > 0 and h > 0:
            return [[round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)] for b in self.boxes]
        return []

    @property
    def labels(self) -> List[str]:
        return [c.label for c in self.objects]

    @property
    def scores(self) -> List[float]:
        return [c.score for c in self.objects]

    @property
    def count(self) -> int:
        return len(self.objects)

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

    @property
    def crops(self) -> Crops:
        """Tự động cắt toàn bộ các vật thể -> Trả về Crops."""
        return Crops(self.objects, self.source_image)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "total_objects": len(self.objects),
            "unique_labels": self.unique_labels,
            "label_counts": self.label_counts,
            "image_size": list(self.source_image.size),
            "speed": self.speed,
        }

    @property
    def has_objects(self) -> bool:
        return len(self.objects) > 0

    @property
    def empty(self) -> bool:
        return len(self.objects) == 0

    @property
    def image_format(self) -> str:
        return getattr(self.source_image, "format", "JPEG") or "JPEG"

    @property
    def image_mode(self) -> str:
        return self.source_image.mode

    @property
    def image_size(self) -> tuple:
        return self.source_image.size

    def filter(self, fn: Callable[[Box], bool]) -> "Detection":
        """Lọc các box bằng Lambda -> Trả về Detection mới."""
        filtered = [c for c in self.objects if fn(c)]
        return Detection(self.source_image, filtered, self.speed, self.image_frame_index, config=self.config)

    def sort(
        self,
        key: Optional[Callable[[Box], Any]] = None,
        reverse: bool = True,
    ) -> "Detection":
        """Sắp xếp các box bằng Lambda."""
        sort_fn = key or (lambda c: c.score)
        return Detection(self.source_image, sorted(self.objects, key=sort_fn, reverse=reverse), self.speed, self.image_frame_index, config=self.config)

    def crop(self, pad: int = 0) -> Crops:
        """Cắt toàn bộ vật thể kèm đệm viền."""
        return Crops(self.objects, self.source_image, pad=pad)

    def plot(
        self,
        line_width: Optional[int] = None,
        labels: bool = True,
        scores: bool = True,
        **kwargs,
    ) -> PIL.Image.Image:
        """Vẽ bounding box và nhãn lên ảnh sử dụng klygo.visual.draw_bboxes."""
        return visual.draw_bboxes(
            image=self.source_image,
            bboxes=self.boxes,
            labels=self.labels if labels else None,
            scores=self.scores if scores else None,
            thickness=line_width or 2,
        )

    draw = plot

    def save(self, output_path: str, line_width: Optional[int] = None, **kwargs) -> None:
        """Vẽ và lưu trực tiếp ảnh nhận diện ra đĩa bằng klygo.media.save."""
        annotated = self.plot(line_width=line_width, **kwargs)
        media.save(output_path, annotated, overwrite=True, verbose=False)

    def show(self, line_width: Optional[int] = None, width: Optional[int] = None, **kwargs) -> None:
        """Vẽ và hiển thị ảnh nhận diện bằng klygo.visual.show_image."""
        annotated = self.plot(line_width=line_width, **kwargs)
        visual.show_image(annotated, width=width)

    def export(self, output_path: str, format: str = "yolo") -> None:
        """Xuất file nhãn YOLO (.txt) hoặc JSON sử dụng klygo.files.save."""
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
            files.save(output_path, "".join(lines), verbose=False)
        elif format.lower() == "json":
            files.save(output_path, self.to_dict(), verbose=False)

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
        if isinstance(index, slice):
            return Detections(self.frames[index], self.source_type, self.fps, self.output_path)
        return self.frames[index]

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        self.frames[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.frames[index]

    def append(self, frame: Detection) -> None:
        self.frames.append(frame)

    def extend(self, other: Union[List[Detection], "Detections"]) -> None:
        items = other.frames if isinstance(other, Detections) else list(other)
        self.frames.extend(items)

    def pop(self, index: int = -1) -> Detection:
        return self.frames.pop(index)

    def insert(self, index: int, frame: Detection) -> None:
        self.frames.insert(index, frame)

    def __len__(self) -> int:
        return len(self.frames)

    def __add__(self, other: "Detections") -> "Detections":
        if not isinstance(other, Detections):
            raise TypeError(f"Không thể cộng Detections với kiểu {type(other)}")
        return Detections(list(self.frames) + list(other.frames), self.source_type, self.fps, self.output_path)

    def __iter__(self):
        return iter(self.frames)

    @property
    def total_objects(self) -> int:
        return sum(len(f) for f in self.frames)

    @property
    def unique_labels(self) -> List[str]:
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
        counts: Dict[str, int] = {}
        for f in self.frames:
            for l in f.labels:
                counts[l] = counts.get(l, 0) + 1
        return counts

    @property
    def labels(self) -> List[List[str]]:
        return [f.labels for f in self.frames]

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def images(self) -> List[PIL.Image.Image]:
        return [f.plot() for f in self.frames]

    def filter(self, fn: Callable[[Any], bool]) -> "Detections":
        filtered = [f for f in self.frames if fn(f)]
        return Detections(filtered, self.source_type, self.fps, self.output_path)

    def sort(
        self,
        key: Optional[Callable[[Detection], Any]] = None,
        reverse: bool = True,
    ) -> "Detections":
        sort_fn = key or (lambda f: len(f))
        return Detections(sorted(self.frames, key=sort_fn, reverse=reverse), self.source_type, self.fps, self.output_path)

    def crop(self, output_path: Optional[str] = None, pad: int = 0) -> Crops:
        all_crops: List[Box] = []
        for f in self.frames:
            all_crops.extend(f.crop(pad=pad).crops)
        crops_obj = Crops(all_crops, pad=pad)
        if output_path:
            crops_obj.export(output_path, format="classification")
        return crops_obj

    def export(self, output_path: str, format: str = "yolo") -> None:
        """Xuất toàn bộ Video thành Flat YOLO Dataset hoặc JSON sử dụng klygo.files và klygo.media."""
        if format.lower() == "yolo":
            img_dir = os.path.join(output_path, "images")
            lbl_dir = os.path.join(output_path, "labels")
            files.mkdir(img_dir)
            files.mkdir(lbl_dir)

            classes = self.unique_labels
            for idx, res in enumerate(self.frames):
                img_name = f"frame_{idx:05d}.jpg"
                lbl_name = f"frame_{idx:05d}.txt"
                media.save(os.path.join(img_dir, img_name), res.source_image, overwrite=True, verbose=False)
                res.export(os.path.join(lbl_dir, lbl_name), format="yolo")

            # Ghi file data.yaml qua klygo.files.save
            yaml_data = {
                "path": os.path.abspath(output_path),
                "train": "images",
                "val": "images",
                "names": {i: name for i, name in enumerate(classes)},
            }
            files.save(os.path.join(output_path, "data.yaml"), yaml_data, verbose=False)
        elif format.lower() == "json":
            files.save(output_path, self.to_dict(), verbose=False)

    def save(self, output_path: str, fps: Optional[float] = None) -> str:
        """Lưu toàn bộ video hoặc thư mục ảnh thành phẩm đã vẽ Bounding Box."""
        target_fps = fps if fps is not None else self.fps
        p_str = str(output_path).lower()
        is_video = p_str.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

        if self.source_type == "video" or is_video:
            final_path = output_path if is_video else os.path.join(output_path, "annotated_video.mp4")
            media.save_video(final_path, self.images, fps=target_fps, overwrite=True, verbose=False)
            self.output_path = final_path
            return final_path
        else:
            files.mkdir(output_path)
            for idx, res in enumerate(self.frames, 1):
                img_path = os.path.join(output_path, f"annotated_{idx:05d}.jpg")
                res.save(img_path)
            self.output_path = output_path
            return output_path

    def draw(self, **kwargs) -> List[PIL.Image.Image]:
        """Vẽ bounding box lên tất cả các frame và trả về danh sách ảnh PIL."""
        return [f.draw(**kwargs) for f in self.frames]

    def show(self, limit: Optional[int] = 5) -> None:
        """Xem trước các frame đầu tiên bằng klygo.visual.show_image."""
        display_items = self.frames if limit is None else self.frames[:limit]
        for res in display_items:
            res.show()

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
