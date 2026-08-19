import os
import inspect
import torch
import PIL.Image
from typing import Any, List, Optional
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo import files, media
from ..interfaces import (
    DetectorModel,
    DetectedObject,
    DetectionResult,
    CroppedObject,
    CropResult,
)
from .. import backends


def _load_source_image(source: Any) -> PIL.Image.Image:
    """
    Đọc và chuẩn hóa duy nhất 1 ảnh đầu vào thông qua klygo.media.load.
    """
    if isinstance(source, PIL.Image.Image):
        return source.convert("RGB")
    elif isinstance(source, str):
        loaded = media.load(source, verbose=False)
        if not loaded:
            raise ValueError(f"Không thể đọc ảnh từ nguồn: {source}")
        return loaded[0].convert("RGB")
    else:
        return media.to_pil(source).convert("RGB")


class GroundingDinoDetect(DetectorModel):
    """
    Trình bao bọc mô hình Zero-shot Object Detection kiến trúc Grounding DINO từ Hugging Face.
    Hỗ trợ nạp Online hoặc Offline trực tiếp từ thư mục trọng số cục bộ.
    """

    def __init__(
        self,
        task: str,
        backend: str,
        num_params: str,
        model_id: str,
        device_map: Optional[str] = None,
        source_model_id: Optional[str] = None,
        half: bool = False,
    ) -> None:
        self.task = task
        self.backend = backend
        self.num_params = num_params
        self.model_id = model_id
        self.source_model_id = source_model_id
        self.half = half

        load_path = model_id
        # Hỗ trợ nạp Offline khi model_id là thư mục cục bộ
        if files.is_dir(model_id) and source_model_id:
            if not files.exists(
                os.path.join(model_id, "model.safetensors")
            ) and not files.exists(os.path.join(model_id, "pytorch_model.bin")):
                load_path = source_model_id

        self.processor = AutoProcessor.from_pretrained(load_path)
        if device_map:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                load_path, device_map=device_map
            )
            self._device = str(self.model.device)
        else:
            self._device = "cpu"
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(load_path)
            if half:
                self.model = self.model.half()
            self.model.to(self._device)

        self.model.eval()

        # Cache trước chữ ký hàm để tăng tốc độ suy luận O(1) mà không tốn chi phí inspect
        sig = inspect.signature(self.processor.post_process_grounded_object_detection)
        self._has_input_ids = "input_ids" in sig.parameters
        self._has_threshold = "threshold" in sig.parameters
        self._has_box_threshold = "box_threshold" in sig.parameters
        self._has_text_threshold = "text_threshold" in sig.parameters

    @property
    def device(self) -> str:
        """Trả về tên thiết bị tính toán hiện tại của mô hình."""
        if hasattr(self.model, "device"):
            return str(self.model.device)
        return self._device

    def to(self, device_name: str) -> "GroundingDinoDetect":
        """
        Tác dụng:
        - Chuyển toàn bộ trọng số mô hình lên thiết bị tính toán được chỉ định ('cpu', 'cuda').
        """
        try:
            self.model.to(device_name)
            self._device = device_name
        except ValueError:
            self._device = str(self.model.device)
        return self

    def predict(
        self,
        source: Any,
        text_prompt: List[str],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Tác dụng:
        - Nhận diện đối tượng trên 1 bức ảnh duy nhất thông qua media.load.

        Đầu vào:
        - source [Any]: 1 bức ảnh (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần tìm kiếm.
        - threshold [float]: Ngưỡng lọc khung giới hạn (Confidence Threshold). Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản. Mặc định: 0.3.

        Đầu ra:
        - [DetectionResult]: Đối tượng kết quả chứa danh sách tọa độ, nhãn và độ tin cậy của ảnh.
        """
        image = _load_source_image(source)

        if not isinstance(text_prompt, list):
            raise TypeError("text_prompt phải là danh sách chuỗi ký tự (list[str]).")

        text_labels = [text_prompt]
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]
        post_kwargs = {
            "outputs": outputs,
            "target_sizes": target_sizes,
        }

        if self._has_input_ids:
            post_kwargs["input_ids"] = inputs.input_ids

        if self._has_threshold:
            post_kwargs["threshold"] = threshold
        else:
            if self._has_box_threshold:
                post_kwargs["box_threshold"] = threshold
            if self._has_text_threshold:
                post_kwargs["text_threshold"] = text_threshold

        results = self.processor.post_process_grounded_object_detection(**post_kwargs)

        result = results[0]
        detected_objects = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            coords = box.tolist()
            detected_objects.append(
                DetectedObject(
                    label=label,
                    score=round(score.item(), 3),
                    box=[round(x, 2) for x in coords],
                    img_size=(image.width, image.height),
                )
            )

        return DetectionResult(source_image=image, objects=detected_objects)

    def crop(
        self,
        source: Any,
        text_prompt: List[str],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> CropResult:
        """
        Tác dụng:
        - Nhận diện và cắt các đối tượng tìm thấy trên 1 bức ảnh duy nhất thành các ảnh con độc lập.

        Đầu vào:
        - source [Any]: 1 bức ảnh (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần cắt.
        - threshold [float]: Ngưỡng lọc khung giới hạn. Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản. Mặc định: 0.3.

        Đầu ra:
        - [CropResult]: Đối tượng tập hợp chứa các ảnh con và siêu dữ liệu tọa độ gốc.
        """
        image = _load_source_image(source)
        pred = self.predict(
            source=image,
            text_prompt=text_prompt,
            threshold=threshold,
            text_threshold=text_threshold,
        )

        cropped_objects = []
        for obj in pred.objects:
            box = obj.box
            xmin = max(0, int(box[0]))
            ymin = max(0, int(box[1]))
            xmax = min(image.width, int(box[2]))
            ymax = min(image.height, int(box[3]))

            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            cropped_objects.append(
                CroppedObject(
                    image=cropped_img,
                    label=obj.label,
                    score=obj.score,
                    box=obj.box,
                    box_normalized=obj.box_normalized,
                )
            )

        return CropResult(source_image=image, crops=cropped_objects)

    def export(self, output_path: str, format: str = "onnx", half: bool = False) -> str:
        """
        Tác dụng:
        - Xuất mô hình sang các kiến trúc tối ưu (ONNX, TensorRT, OpenVINO, FP16) kèm config.json.
        """
        output_path = os.path.abspath(output_path)
        files.mkdir(output_path)
        format = format.lower().lstrip(".")

        if format == "onnx":
            backends.export_onnx(self.model_id, self.processor, output_path, half=half)
        elif format in ["engine", "tensorrt", "trt"]:
            backends.export_tensorrt(self.model_id, self.processor, output_path, half=half)
        elif format in ["openvino", "xml", "ov"]:
            backends.export_openvino(self.model_id, self.processor, output_path, half=half)
        elif format in ["torchscript", "pt", "safetensors", "torch", "fp16"]:
            backends.export_torch(self.model_id, self.processor, output_path, half=half)

        config_data = {
            "class": "GroundingDinoDetect",
            "task": self.task,
            "backend": format,
            "num_params": self.num_params,
            "model_id": output_path,
            "source_model_id": self.model_id,
            "half": half,
        }
        config_file = os.path.join(output_path, "config.json")
        files.save(config_file, config_data, overwrite=True, verbose=False)

        return output_path

    def dataset(
        self,
        output_path: str,
        format: str,
        source: Any,
        text_prompt: List[str],
        batch_size: int = 16,
        threshold: float = 0.4,
    ) -> None:
        """
        Tác dụng:
        - Tự động tạo bộ dữ liệu huấn luyện định dạng YOLO hoặc Classification từ nguồn ảnh/video.

        Đầu vào:
        - output_path [str]: Thư mục lưu trữ bộ dữ liệu đầu ra.
        - format [str]: Định dạng xuất ('yolo' hoặc 'classification').
        - source [str | List]: Đường dẫn thư mục ảnh, file video, hoặc danh sách ảnh đã đọc sẵn từ media.load.
        - text_prompt [List[str]]: Danh sách các lớp nhãn đối tượng cần trích xuất.
        - batch_size [int]: Kích thước xử lý theo lô. Mặc định: 16.
        - threshold [float]: Ngưỡng độ tin cậy nhận diện. Mặc định: 0.4.
        """
        from klygo.datasets import detect

        detect.export(
            model=self,
            output_path=output_path,
            format=format,
            source=source,
            text_prompt=text_prompt,
            batch_size=batch_size,
            threshold=threshold,
        )

    def warmup(self) -> None:
        """Khởi động mô hình với dữ liệu giả lập."""
        dummy_img = PIL.Image.new("RGB", (1, 1), color="black")
        try:
            self.predict(source=dummy_img, text_prompt=["dummy"], threshold=0.9, text_threshold=0.9)
        except Exception:
            pass

    def unload(self) -> None:
        """Giải phóng mô hình khỏi GPU và dọn sạch bộ nhớ đệm VRAM."""
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def help(self) -> None:
        """In ra thông tin mô hình và danh sách các hàm nghiệp vụ."""
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print("=" * 52)
        print("1. predict(source, text_prompt, threshold=0.4, text_threshold=0.3)")
        print("   Nhan dien doi tuong tren 1 anh (Path, PIL, NumPy, Tensor).")
        print("2. crop(source, text_prompt, threshold=0.4, text_threshold=0.3)")
        print("   Cat doi tuong thanh danh sach anh con PIL Images.")
        print("3. dataset(output_path, format, source, text_prompt, batch_size=16, threshold=0.4)")
        print("   Tao dataset YOLO hoac Classification tu thu muc anh, video, hoac List[PIL.Image].")
        print("4. export(output_path, format='onnx', half=False)")
        print("   Xuat mo hinh sang ONNX, TensorRT, OpenVINO, hoac FP16.")

