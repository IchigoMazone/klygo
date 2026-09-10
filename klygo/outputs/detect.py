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
from typing import List, Dict, Any, Optional, Union, Callable, Iterable
import PIL.Image
import numpy as np

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
        self._source_image = None
        self._source_path = None
        self._url = None
        self.objects = objects  # Danh sách các Box
        self.source_image = source_image
        self.speed = speed or {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0, "total": 0.0}
        self.image_frame_index = image_frame_index
        self.frame_index = image_frame_index
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Cấu hình động tự do (Open Configuration)
        self.config: Dict[str, Any] = dict(config or {})
        self.config.update(kwargs)

    @property
    def source_image(self) -> Any:
        """Ảnh gốc tương ứng với kết quả nhận diện (hỗ trợ đọc hoặc gán lại ảnh mới)."""
        return self._source_image

    @source_image.setter
    def source_image(self, img: Any) -> None:
        self._source_image = img
        raw_path = getattr(img, "path", None)
        if raw_path is None:
            raw_path = getattr(img, "filename", None)
        
        if raw_path is not None:
            self._source_path = str(raw_path)
            self._url = self._source_path
        if hasattr(self, "objects"):
            for c in self.objects:
                c.parent_image = img

    @property
    def url(self) -> Optional[str]:
        """URL hoặc đường dẫn tệp ảnh gốc."""
        return self._url

    @url.setter
    def url(self, new_url: Union[str, os.PathLike]) -> None:
        if not isinstance(new_url, (str, os.PathLike)):
            raise TypeError(f"Replacement URL must be a valid file path or URL string, got {type(new_url)}")
        self._url = str(new_url)
        self._source_path = self._url
        # Tự động nạp Zero-RAM LazyImage từ URL mới và đồng bộ parent_image
        self.source_image = media.LazyImage(self._url)

    @property
    def source_path(self) -> Optional[str]:
        """Đường dẫn tệp ảnh gốc."""
        return self._source_path

    @source_path.setter
    def source_path(self, new_path: Optional[Union[str, os.PathLike]]) -> None:
        if new_path is None:
            self._source_path = None
            self._url = None
        else:
            self.url = new_path

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dictionary metadata và tham số cấu hình của frame."""
        return self.config

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

    def to_array(self, bgr: bool = False, annotated: bool = True, **kwargs) -> np.ndarray:
        """
        Chuyển đổi ảnh kết quả nhận diện sang mảng NumPy ndarray.
        - bgr=True: Chuyển sang hệ màu BGR chuẩn OpenCV để hiển thị trực tiếp bằng cv2.imshow.
        - annotated=True (mặc định): Trả về ảnh đã vẽ Bounding Box.
        - annotated=False: Trả về ảnh gốc dạng ndarray.
        """
        img = self.plot(**kwargs) if annotated else self.source_image
        if hasattr(img, "to_array"):
            arr = img.to_array()
        else:
            arr = np.array(img)
        if bgr:
            return arr[:, :, ::-1]
        return arr

    def to_numpy(self, bgr: bool = False, annotated: bool = True, **kwargs) -> np.ndarray:
        """Alias cho to_array()."""
        return self.to_array(bgr=bgr, annotated=annotated, **kwargs)

    def to_bgr(self, annotated: bool = True, **kwargs) -> np.ndarray:
        """Trả về mảng NumPy hệ màu BGR để hiển thị trực tiếp bằng cv2.imshow."""
        return self.to_array(bgr=True, annotated=annotated, **kwargs)

    def __array__(self, dtype=None) -> np.ndarray:
        """Hỗ trợ tự động chuyển đổi khi gọi np.array(det)."""
        arr = self.to_array(annotated=True)
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def save(self, output_path: str, line_width: Optional[int] = None, **kwargs) -> None:
        """Vẽ và lưu trực tiếp ảnh nhận diện ra đĩa bằng klygo.media.save."""
        annotated = self.plot(line_width=line_width, **kwargs)
        media.save(output_path, annotated, overwrite=True, verbose=False)

    def show(self, line_width: Optional[int] = None, width: Optional[int] = None, **kwargs) -> None:
        """Vẽ và hiển thị ảnh nhận diện bằng klygo.visual.show_image."""
        annotated = self.plot(line_width=line_width, **kwargs)
        visual.show_image(annotated, width=width)

    def export(self, output_path: str, format: str = "yolo", classes: Optional[List[str]] = None) -> None:
        """Xuất file nhãn YOLO (.txt) hoặc JSON sử dụng klygo.files.save."""
        if format.lower() == "yolo":
            w, h = self.source_image.size
            unique_classes = classes if classes is not None else sorted(list(set(self.labels)))
            lines = []
            for c in self.objects:
                if c.label not in unique_classes:
                    continue
                cid = unique_classes.index(c.label)
                cx = (c.box[0] + c.box[2]) / 2.0 / w
                cy = (c.box[1] + c.box[3]) / 2.0 / h
                bw = (c.box[2] - c.box[0]) / w
                bh = (c.box[3] - c.box[1]) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            from klygo import files
            files.save(output_path, "".join(lines), verbose=False)
        elif format.lower() == "json":
            from klygo import files
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
        if self.source_path is not None:
            d["source_path"] = self.source_path
            d["url"] = self.url
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
    Hỗ trợ đồng nhất cả chế độ In-Memory (danh sách) lẫn Stream (tiết kiệm RAM chống tràn bộ nhớ).
    """

    def __init__(
        self,
        frames: Union[List[Detection], Iterable[Detection]],
        source_type: str = "video",
        fps: float = 30.0,
        output_path: Optional[str] = None,
        stream: bool = False,
        total_frames: Optional[int] = None,
    ) -> None:
        self.is_stream = bool(stream) or (not isinstance(frames, (list, tuple)) and hasattr(frames, "__iter__"))
        self.source_type = source_type
        self.fps = float(fps) if fps else 30.0
        self.output_path = output_path
        self.total_frames = total_frames
        self._stream_cache: List[Detection] = []

        if self.is_stream:
            self._iterator = iter(frames) if frames is not None else iter(())
            self._frames: List[Detection] = []
        else:
            self._frames = list(frames) if frames is not None else []
            self._iterator = None
            if self.total_frames is None:
                self.total_frames = len(self._frames)

    @property
    def frames(self) -> List[Detection]:
        if self.is_stream:
            return self.to_list()
        return self._frames

    @frames.setter
    def frames(self, value: Any) -> None:
        if self.is_stream and not isinstance(value, (list, tuple)):
            self._iterator = iter(value)
        else:
            self._frames = list(value) if value is not None else []
            self.is_stream = False
            self.total_frames = len(self._frames)

    @property
    def results(self) -> List[Detection]:
        return self.frames

    @results.setter
    def results(self, value: Any) -> None:
        self.frames = value

    def __getitem__(self, index: Union[int, slice]) -> Union[Detection, "Detections"]:
        if self.is_stream:
            if isinstance(index, int):
                if index < 0:
                    raise IndexError("Negative indexing is not supported in streaming Detections.")
                while len(self._stream_cache) <= index:
                    try:
                        self._stream_cache.append(next(self._iterator))
                    except StopIteration:
                        raise IndexError("Detections stream index out of range")
                return self._stream_cache[index]
            elif isinstance(index, slice):
                start, stop, step = index.start, index.stop, index.step
                step = 1 if step is None else step
                if step <= 0:
                    raise ValueError("Slice step must be positive for streaming Detections")

                if (start is not None and start < 0) or (stop is not None and stop < 0):
                    raise ValueError("Negative slice indices are not supported on streaming Detections. Use to_list() first.")

                def _slice_gen():
                    idx = 0
                    while True:
                        if start is not None and idx < start:
                            try:
                                if idx < len(self._stream_cache):
                                    _ = self._stream_cache[idx]
                                else:
                                    _ = next(self._iterator)
                                idx += 1
                                continue
                            except StopIteration:
                                break

                        if stop is not None and idx >= stop:
                            break

                        offset = idx if start is None else (idx - start)
                        if offset % step == 0:
                            try:
                                if idx < len(self._stream_cache):
                                    det = self._stream_cache[idx]
                                else:
                                    det = next(self._iterator)
                                    self._stream_cache.append(det)
                                yield det
                            except StopIteration:
                                break
                        else:
                            try:
                                if idx < len(self._stream_cache):
                                    _ = self._stream_cache[idx]
                                else:
                                    det = next(self._iterator)
                                    self._stream_cache.append(det)
                            except StopIteration:
                                break
                        idx += 1

                new_total = None
                if self.total_frames is not None:
                    s_start = 0 if start is None else min(start, self.total_frames)
                    s_stop = self.total_frames if stop is None else min(stop, self.total_frames)
                    if s_stop > s_start:
                        new_total = (s_stop - s_start + step - 1) // step
                    else:
                        new_total = 0

                return Detections(
                    _slice_gen(),
                    self.source_type,
                    self.fps / step if step > 1 else self.fps,
                    self.output_path,
                    stream=True,
                    total_frames=new_total,
                )
            raise TypeError(f"Invalid index type: {type(index)}")

        if isinstance(index, slice):
            return Detections(self._frames[index], self.source_type, self.fps, self.output_path)
        return self._frames[index]

    def map(self, func: Callable[[Detection], Any]) -> "Detections":
        """
        Áp dụng hàm biến đổi func lên từng Detection frame.
        Hỗ trợ cả In-Memory lẫn Stream.
        """
        if self.is_stream:
            def _map_gen():
                for f in self:
                    yield func(f)
            return Detections(_map_gen(), self.source_type, self.fps, self.output_path, stream=True, total_frames=self.total_frames)
        mapped = [func(f) for f in self._frames]
        return Detections(mapped, self.source_type, self.fps, self.output_path)

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        if self.is_stream:
            self.to_list()
        self._frames[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        if self.is_stream:
            self.to_list()
        del self._frames[index]

    def append(self, frame: Detection) -> None:
        if self.is_stream:
            self.to_list()
        self._frames.append(frame)
        self.total_frames = len(self._frames)

    def extend(self, other: Union[List[Detection], "Detections"]) -> None:
        if self.is_stream:
            self.to_list()
        items = other.frames if isinstance(other, Detections) else list(other)
        self._frames.extend(items)
        self.total_frames = len(self._frames)

    def pop(self, index: int = -1) -> Detection:
        if self.is_stream:
            self.to_list()
        res = self._frames.pop(index)
        self.total_frames = len(self._frames)
        return res

    def insert(self, index: int, frame: Detection) -> None:
        if self.is_stream:
            self.to_list()
        self._frames.insert(index, frame)
        self.total_frames = len(self._frames)

    def __len__(self) -> int:
        if self.is_stream:
            return self.total_frames if self.total_frames is not None else len(self._stream_cache)
        return len(self._frames)

    def to_list(self) -> List[Detection]:
        """Chuyển toàn bộ các frame của stream thành list chuẩn trong bộ nhớ."""
        if self.is_stream:
            items = list(self)
            self._frames = items
            self.is_stream = False
            self.total_frames = len(items)
            return items
        return list(self._frames)

    def __add__(self, other: "Detections") -> "Detections":
        if not isinstance(other, Detections):
            raise TypeError(f"Không thể cộng Detections với kiểu {type(other)}")
        return Detections(list(self) + list(other), self.source_type, self.fps, self.output_path)

    def __iter__(self):
        if self.is_stream:
            idx = 0
            while True:
                if idx < len(self._stream_cache):
                    yield self._stream_cache[idx]
                else:
                    try:
                        item = next(self._iterator)
                        self._stream_cache.append(item)
                        yield item
                    except StopIteration:
                        break
                idx += 1
        else:
            yield from iter(self._frames)

    @property
    def total_objects(self) -> int:
        target = self._frames if not self.is_stream else self._stream_cache
        return sum(len(f) for f in target)

    @property
    def unique_labels(self) -> List[str]:
        seen = set()
        out = []
        target = self._frames if not self.is_stream else self._stream_cache
        for f in target:
            for l in f.labels:
                if l not in seen:
                    seen.add(l)
                    out.append(l)
        return out

    @property
    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        target = self._frames if not self.is_stream else self._stream_cache
        for f in target:
            for l in f.labels:
                counts[l] = counts.get(l, 0) + 1
        return counts

    @property
    def boxes(self) -> Any:
        """Danh sách bounding boxes. Với ảnh đơn trả về list các box [x1, y1, x2, y2]."""
        if len(self) == 1:
            return self[0].boxes
        target = self._frames if not self.is_stream else self._stream_cache
        return [f.boxes for f in target]

    @property
    def labels(self) -> Any:
        """Danh sách nhãn. Với ảnh đơn trả về list các nhãn string."""
        if len(self) == 1:
            return self[0].labels
        target = self._frames if not self.is_stream else self._stream_cache
        return [f.labels for f in target]

    @property
    def scores(self) -> Any:
        """Danh sách điểm tin cậy. Với ảnh đơn trả về list các score float."""
        if len(self) == 1:
            return self[0].scores
        target = self._frames if not self.is_stream else self._stream_cache
        return [f.scores for f in target]

    @property
    def url(self) -> Optional[str]:
        """URL của ảnh (với kết quả ảnh đơn) hoặc đường dẫn xuất video."""
        if len(self) == 1:
            return self[0].url
        return self.output_path

    @url.setter
    def url(self, new_url: Union[str, os.PathLike]) -> None:
        if len(self) == 1:
            self[0].url = new_url
        else:
            raise ValueError("Không thể gán 1 URL duy nhất cho nhiều frame kết quả. Hãy gán url cho từng frame results[i].url = 'path'")

    @property
    def count(self) -> int:
        return len(self)

    @property
    def images(self) -> List[PIL.Image.Image]:
        """CẢNH BÁO: Tạo list toàn bộ ảnh render. Sẽ tốn RAM nếu video quá dài!"""
        target = self.to_list() if self.is_stream else self._frames
        return [f.plot() for f in target]

    def iter_images(self, **kwargs):
        """Generator sinh ảnh đã vẽ Box từng frame một (chống tràn RAM)."""
        for f in self:
            yield f.plot(**kwargs)

    def filter(self, fn: Callable[[Any], bool]) -> "Detections":
        if self.is_stream:
            def _gen():
                for f in self:
                    if fn(f):
                        yield f
            return Detections(_gen(), self.source_type, self.fps, self.output_path, stream=True)
        filtered = [f for f in self._frames if fn(f)]
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
        from klygo import files, media
        if format.lower() == "yolo":
            img_dir = os.path.join(output_path, "images")
            lbl_dir = os.path.join(output_path, "labels")
            files.mkdir(img_dir)
            files.mkdir(lbl_dir)

            classes = sorted(self.unique_labels)
            for idx, res in enumerate(self.frames):
                img_name = f"frame_{idx:05d}.jpg"
                lbl_name = f"frame_{idx:05d}.txt"
                media.save(os.path.join(img_dir, img_name), res.source_image, overwrite=True, verbose=False)
                res.export(os.path.join(lbl_dir, lbl_name), format="yolo", classes=classes)

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

    def save(self, output_path: str, fps: Optional[float] = None, **kwargs) -> str:
        """Lưu toàn bộ video hoặc thư mục ảnh thành phẩm đã vẽ Bounding Box."""
        target_fps = fps if fps is not None else self.fps
        p_str = str(output_path).lower()
        is_video = p_str.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

        if self.source_type == "video" or is_video:
            final_path = output_path if is_video else os.path.join(output_path, "annotated_video.mp4")
            # Sử dụng iter_images (Generator) thay vì self.images (List) để chống OOM RAM.
            media.save_video(final_path, self.iter_images(**kwargs), fps=target_fps, overwrite=True, verbose=False)
            self.output_path = final_path
            return final_path
        else:
            files.mkdir(output_path)
            for idx, res in enumerate(self.frames, 1):
                img_path = os.path.join(output_path, f"annotated_{idx:05d}.jpg")
                res.save(img_path, **kwargs)
            self.output_path = output_path
            return output_path

    def draw(self, **kwargs) -> List[PIL.Image.Image]:
        """Vẽ bounding box lên tất cả các frame và trả về danh sách ảnh PIL."""
        return [f.draw(**kwargs) for f in self.frames]

    def plot(self, **kwargs) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """Vẽ bounding box lên ảnh. Với ảnh đơn trả về PIL Image, với video/danh sách trả về List."""
        if len(self) == 1:
            return self[0].plot(**kwargs)
        return [f.plot(**kwargs) for f in self.frames]

    def to_array(self, bgr: bool = False, annotated: bool = True, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Chuyển đổi ảnh thành mảng NumPy ndarray.
        - Với ảnh đơn: Trả về trực tiếp mảng np.ndarray dạng [H, W, C].
        - Với nhiều frame/video: Trả về danh sách các mảng np.ndarray.
        - bgr=True: Chuyển sang hệ màu BGR chuẩn OpenCV để đưa thẳng vào cv2.imshow / cv2.imwrite.
        - annotated=True (mặc định): Trả về ảnh đã vẽ Bounding Box.
        - annotated=False: Trả về ảnh gốc dạng ndarray.
        """
        if len(self) == 1:
            return self[0].to_array(bgr=bgr, annotated=annotated, **kwargs)
        return [f.to_array(bgr=bgr, annotated=annotated, **kwargs) for f in self]

    def to_numpy(self, bgr: bool = False, annotated: bool = True, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Alias cho to_array()."""
        return self.to_array(bgr=bgr, annotated=annotated, **kwargs)

    def to_bgr(self, annotated: bool = True, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Trả về mảng NumPy hệ màu BGR chuẩn OpenCV để hiển thị trực tiếp bằng cv2.imshow."""
        return self.to_array(bgr=True, annotated=annotated, **kwargs)

    def __array__(self, dtype=None) -> np.ndarray:
        """Hỗ trợ tự động chuyển đổi khi gọi np.array(results)."""
        if len(self) == 1:
            return self[0].__array__(dtype=dtype)
        arr_list = [f.to_array(annotated=True) for f in self]
        try:
            res = np.array(arr_list)
        except Exception:
            res = np.empty(len(arr_list), dtype=object)
            res[:] = arr_list
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def show(self, limit: Optional[int] = 5) -> None:
        """Xem trước các frame đầu tiên bằng klygo.visual.show_image."""
        if self.is_stream:
            display_items = []
            max_cnt = limit if limit is not None else 5
            for i in range(max_cnt):
                try:
                    display_items.append(self[i])
                except (IndexError, StopIteration):
                    break
        else:
            display_items = self._frames if limit is None else self._frames[:limit]
        for res in display_items:
            res.show()

    def to_dict(self) -> Dict[str, Any]:
        """Xuất cấu trúc Key-Value chuẩn toàn bộ video."""
        return {
            "source_type": self.source_type,
            "total_frames": len(self),
            "fps": self.fps,
            "total_objects": self.total_objects,
            "unique_labels": self.unique_labels,
            "label_counts": self.label_counts,
            "frames": [f.to_dict() for f in self],
        }

    def __repr__(self) -> str:
        if self.is_stream:
            cnt_str = f"{self.total_frames} frames" if self.total_frames is not None else "streaming"
            return f"<Detections (Stream): source_type='{self.source_type}', total={cnt_str}, fps={self.fps}>"
        return f"<Detections frames={len(self._frames)}, total_objects={self.total_objects}, unique_labels={self.unique_labels}>"


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
