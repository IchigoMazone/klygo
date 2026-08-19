import os
import PIL.Image
from typing import List, Dict, Any, Optional
from klygo import files, media


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

    def plot(self, outline_color: str = "red", width: int = 3) -> PIL.Image.Image:
        """
        Tác dụng:
        - Vẽ bounding box và tên nhãn trực tiếp lên ảnh gốc và trả về đối tượng ảnh PIL.Image.

        Đầu vào:
        - outline_color [str]: Màu viền khung chữ nhật (Mặc định: 'red').
        - width [int]: Độ dày đường viền (Mặc định: 3).

        Đầu ra:
        - [PIL.Image.Image]: Bức ảnh mới đã được vẽ đóng khung nhãn.
        """
        from PIL import ImageDraw

        annotated_image = self.source_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for obj in self.objects:
            draw.rectangle(obj.box, outline=outline_color, width=width)
            text = f"{obj.label} ({obj.score:.2f})"
            draw.text((obj.xmin + 5, obj.ymin + 5), text, fill=outline_color)

        return annotated_image

    def save(self, output_path: str, outline_color: str = "red", width: int = 3) -> None:
        """Vẽ và lưu trực tiếp ảnh nhận diện ra đường dẫn đĩa bằng klygo.media.save."""
        annotated = self.plot(outline_color=outline_color, width=width)
        media.save(output_path, annotated, overwrite=True, verbose=False)

    def show(self, outline_color: str = "red", width: int = 3) -> "PIL.Image.Image":
        """Vẽ và hiển thị ảnh nhận diện (tương thích Desktop, Colab và Jupyter Notebook)."""
        annotated = self.plot(outline_color=outline_color, width=width)
        try:
            from IPython.display import display
            display(annotated)
        except ImportError:
            annotated.show()
        return annotated

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

    def show(self) -> None:
        """Mở xem ảnh con ngay trên màn hình."""
        self.image.show()

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

    def to_dict(self) -> Dict[str, Any]:
        """Xuất danh sách siêu dữ liệu của tất cả các ảnh con ra dạng Dictionary."""
        return {
            "crops": [crop.to_dict() for crop in self.crops],
            "total": len(self.crops),
        }

    def __len__(self) -> int:
        return len(self.crops)

    def __iter__(self):
        return iter(self.crops)

    def __getitem__(self, index):
        return self.crops[index]

    def __repr__(self) -> str:
        summary = f"CropResult: {len(self.crops)} cropped objects"
        if len(self.crops) > 0:
            details = ", ".join([f"{c.label} ({c.image.size[0]}x{c.image.size[1]})" for c in self.crops[:5]])
            if len(self.crops) > 5:
                details += ", ..."
            summary += f" [{details}]"
        return summary
