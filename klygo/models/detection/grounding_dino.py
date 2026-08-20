import os
import logging
import warnings
import inspect
import torch
import PIL.Image
from typing import Any, List, Optional, Dict, Union

# Tắt các cảnh báo không cần thiết từ Hugging Face Hub và Transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=".*The key `labels` is will return integer ids.*")
warnings.filterwarnings("ignore", message=".*text_labels.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

try:
    import huggingface_hub.utils.logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

try:
    import transformers.utils.logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

for _logger_name in ["huggingface_hub", "huggingface_hub.utils._http", "transformers"]:
    try:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
    except Exception:
        pass

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
        int8: bool = False,
        **kwargs,
    ) -> None:
        self.task = task
        self.backend = backend
        self.num_params = num_params
        self.model_id = model_id
        self.source_model_id = source_model_id
        self.half = half
        self.int8 = int8

        load_path = model_id
        # Hỗ trợ nạp Offline khi model_id là thư mục cục bộ
        if files.is_dir(model_id) and source_model_id:
            if not files.exists(
                os.path.join(model_id, "model.safetensors")
            ) and not files.exists(os.path.join(model_id, "pytorch_model.bin")):
                load_path = source_model_id

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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
                elif self._device == "cpu":
                    try:
                        if next(self.model.parameters()).dtype == torch.float16:
                            self.model = self.model.float()
                    except Exception:
                        pass
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
            # Nếu chuyển về CPU mà mô hình đang là float16, tự động chuyển về float32 để CPU tính toán được
            if device_name == "cpu" and next(self.model.parameters()).dtype == torch.float16:
                self.model = self.model.float()
            elif "cuda" in device_name and self.half:
                self.model = self.model.half()
        except ValueError:
            self._device = str(self.model.device)
        return self

    def predict(
        self,
        source: Any,
        text_prompt: Optional[List[str]] = None,
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        data: Optional[str] = None,
    ) -> DetectionResult:
        """
        Tác dụng:
        - Nhận diện đối tượng trên 1 bức ảnh duy nhất thông qua media.load.

        Đầu vào:
        - source [Any]: 1 bức ảnh (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần tìm kiếm (hoặc nạp từ data='data.yaml').
        - threshold [float]: Ngưỡng lọc khung giới hạn (Confidence Threshold). Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản. Mặc định: 0.3.
        - data [str]: Đường dẫn file data.yaml (tự động lấy danh sách nhãn lớp giống YOLO).

        Đầu ra:
        - [DetectionResult]: Đối tượng kết quả chứa danh sách tọa độ, nhãn, độ tin cậy và tốc độ suy luận.
        """
        import time
        from ..interfaces.base import _parse_data_yaml

        if data:
            _, yaml_names = _parse_data_yaml(data)
            text_prompt = text_prompt or yaml_names

        if not text_prompt or not isinstance(text_prompt, list):
            raise TypeError("text_prompt phải là danh sách chuỗi ký tự (list[str]) hoặc nạp qua data='data.yaml'.")

        t0 = time.perf_counter()
        image = _load_source_image(source)

        text_labels = [text_prompt]
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt")

        # Xác định thiết bị tính toán
        model_device = getattr(self.model, "device", self._device)
        is_cuda = "cuda" in str(model_device)

        # Trên CPU: PyTorch MSDeformAttn grid_sample không hỗ trợ float16, tự động chuyển model sang float32
        if not is_cuda and next(self.model.parameters()).dtype == torch.float16:
            self.model = self.model.float()

        model_dtype = next(self.model.parameters()).dtype
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(device=model_device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(device=model_device)

        t1 = time.perf_counter()

        with torch.no_grad():
            if is_cuda and self.half:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        if is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        t2 = time.perf_counter()

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

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            results = self.processor.post_process_grounded_object_detection(**post_kwargs)

        result = results[0]
        detected_objects = []
        labels_list = result.get("text_labels", result.get("labels", []))
        for box, score, label in zip(result["boxes"], result["scores"], labels_list):
            coords = box.tolist()
            detected_objects.append(
                DetectedObject(
                    label=str(label),
                    score=round(score.item(), 3),
                    box=[round(x, 2) for x in coords],
                    img_size=(image.width, image.height),
                )
            )

        t3 = time.perf_counter()

        speed = {
            "preprocess": round((t1 - t0) * 1000, 2),
            "inference": round((t2 - t1) * 1000, 2),
            "postprocess": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        }

        return DetectionResult(source_image=image, objects=detected_objects, speed=speed)

    def crop(
        self,
        source: Any,
        text_prompt: Optional[List[str]] = None,
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        data: Optional[str] = None,
    ) -> CropResult:
        """
        Tác dụng:
        - Nhận diện và cắt các đối tượng tìm thấy trên 1 bức ảnh duy nhất thành các ảnh con độc lập.

        Đầu vào:
        - source [Any]: 1 bức ảnh (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần cắt.
        - threshold [float]: Ngưỡng lọc khung giới hạn. Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản. Mặc định: 0.3.
        - data [str]: Đường dẫn file data.yaml (tự động lấy danh sách nhãn lớp giống YOLO).

        Đầu ra:
        - [CropResult]: Đối tượng tập hợp chứa các ảnh con và siêu dữ liệu tọa độ gốc.
        """
        image = _load_source_image(source)
        pred = self.predict(
            source=image,
            text_prompt=text_prompt,
            threshold=threshold,
            text_threshold=text_threshold,
            data=data,
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

    def export(
        self,
        output_path: str,
        format: str = "onnx",
        half: bool = False,
        int8: bool = False,
        data: Optional[str] = None,
        calibration_source: Optional[Union[str, List[Any]]] = None,
        calibration_prompts: Optional[List[str]] = None,
    ) -> str:
        """
        Tác dụng:
        - Xuất mô hình sang các kiến trúc tối ưu (ONNX, TensorRT, OpenVINO, FP16, INT8) kèm config.json.
        """
        from ..interfaces.base import _parse_data_yaml

        if data:
            d_source, d_names = _parse_data_yaml(data)
            calibration_source = calibration_source or d_source
            calibration_prompts = calibration_prompts or d_names

        output_path = os.path.abspath(output_path)
        files.mkdir(output_path)
        format = format.lower().lstrip(".")

        if format == "onnx":
            backends.export_onnx(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )
        elif format in ["engine", "tensorrt", "trt"]:
            backends.export_tensorrt(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )
        elif format in ["openvino", "xml", "ov"]:
            backends.export_openvino(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )
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
            "int8": int8,
        }
        klygo_file = os.path.join(output_path, "klygo.json")
        files.save(klygo_file, config_data, overwrite=True, verbose=False)

        return output_path

    def dataset(
        self,
        output_path: str,
        format: str,
        source: Optional[Union[str, List[Any]]] = None,
        text_prompt: Optional[List[str]] = None,
        data: Optional[str] = None,
        batch_size: int = 16,
        threshold: float = 0.4,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Tác dụng:
        - Tự động tạo bộ dữ liệu huấn luyện định dạng Detection hoặc Classification từ nguồn ảnh/video.

        Đầu vào:
        - output_path [str]: Thư mục lưu trữ bộ dữ liệu đầu ra.
        - format [str]: Định dạng xuất ('detection' hoặc 'classification').
        - source [str | List]: Đường dẫn thư mục ảnh, file video, hoặc danh sách ảnh đã đọc sẵn từ media.load.
        - text_prompt [List[str]]: Danh sách các lớp nhãn đối tượng cần trích xuất.
        - data [str]: File cấu hình data.yaml để nạp tự động ảnh và nhãn.
        - batch_size [int]: Kích thước xử lý theo lô. Mặc định: 16.
        - threshold [float]: Ngưỡng độ tin cậy nhận diện. Mặc định: 0.4.
        - verbose [bool]: Hiển thị thanh tiến trình. Mặc định: True.
        """
        from ..interfaces.base import _parse_data_yaml
        from klygo.datasets import detect

        if data:
            d_source, d_names = _parse_data_yaml(data)
            source = source or d_source
            text_prompt = text_prompt or d_names

        detect.export(
            model=self,
            output_path=output_path,
            format=format,
            source=source,
            text_prompt=text_prompt,
            batch_size=batch_size,
            threshold=threshold,
            verbose=verbose,
            **kwargs,
        )

    def benchmark(
        self,
        source: Optional[Any] = None,
        text_prompt: Optional[List[str]] = None,
        data: Optional[str] = None,
        iterations: int = 20,
        warmup: int = 5,
        threshold: float = 0.4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Tác dụng:
        - Đo đạc và chấm điểm đánh giá tốc độ suy luận (Độ trễ Latency ms / Tốc độ FPS) của mô hình.

        Đầu vào:
        - data [str]: Đường dẫn file data.yaml (Tự động nạp danh sách ảnh và nhãn giống YOLO).
        - source [Any]: Ảnh thử nghiệm (mặc định tạo ảnh chuẩn 640x640 nếu là None).
        - text_prompt [List[str]]: Danh sách nhãn từ khóa cần đo (Mặc định: ['object']).
        - iterations [int]: Số lần lặp đo đạc. Mặc định: 20.
        - warmup [int]: Số lần chạy khởi động trước khi bấm giờ. Mặc định: 5.
        - threshold [float]: Ngưỡng độ tin cậy nhận diện. Mặc định: 0.4.
        - verbose [bool]: In bảng báo cáo chi tiết ra màn hình console. Mặc định: True.

        Đầu ra:
        - [dict]: Kết quả đo đạc gồm latency_avg_ms, latency_min_ms, latency_max_ms, fps, device, backend...
        """
        import time
        from ..interfaces.base import _parse_data_yaml

        if data:
            d_source, d_names = _parse_data_yaml(data)
            source = source or d_source
            text_prompt = text_prompt or d_names

        img = source if source is not None else PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        prompts = text_prompt or ["object"]

        # 1. Warmup
        for _ in range(warmup):
            self.predict(img, text_prompt=prompts, threshold=threshold)

        # 2. Benchmark
        latencies = []
        is_cuda = "cuda" in str(getattr(self.model, "device", self._device))

        for _ in range(iterations):
            t_start = time.perf_counter()
            self.predict(img, text_prompt=prompts, threshold=threshold)
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

        w_dim, h_dim = (img.width, img.height) if isinstance(img, PIL.Image.Image) else (640, 640)

        report = {
            "model_id": self.model_id,
            "backend": self.backend,
            "device": self.device,
            "image_size": f"{w_dim}x{h_dim}",
            "iterations": iterations,
            "warmup": warmup,
            "latency_avg_ms": round(avg_latency * 1000, 2),
            "latency_min_ms": round(min_latency * 1000, 2),
            "latency_max_ms": round(max_latency * 1000, 2),
            "fps": round(fps, 1),
        }

        if verbose:
            print("=" * 60)
            print("         BAO CAO DANH GIA HIEU NANG & TOC DO MO HINH")
            print("=" * 60)
            print(f" * Mo hinh      : {report['model_id']}")
            print(f" * Backend      : {report['backend']}")
            print(f" * Thiet bi     : {report['device']}")
            print(f" * Kich thuoc   : {report['image_size']}")
            print(f" * So vong lap  : {report['iterations']} (Warmup: {report['warmup']})")
            print("-" * 60)
            print(f" * Do tre TB    : {report['latency_avg_ms']} ms / anh")
            print(f" * Nhanh nhat   : {report['latency_min_ms']} ms")
            print(f" * Cham nhat    : {report['latency_max_ms']} ms")
            print(f" * Toc do (FPS) : {report['fps']} FPS (frames / sec)")
            print("=" * 60)

        return report

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
        print("   Tao dataset 'detection' hoac 'classification' tu thu muc anh, video, hoac List[PIL.Image].")
        print("4. export(output_path, format='safetensors', half=False)")
        print("   Xuat mo hinh sang SafeTensors FP16, ONNX, TensorRT, OpenVINO.")
        print("5. benchmark(source=None, iterations=20, warmup=5)")
        print("   Cham diem danh gia toc do suy luan (Do tre Latency ms / Toc do FPS).")

