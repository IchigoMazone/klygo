"""
Lớp cơ sở trừu tượng (`Detector`) định nghĩa chuẩn giao diện chung cho tất cả các mô hình nhận diện đối tượng trong Klygo.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import PIL.Image

from klygo.models import utils
from klygo.outputs.detect import Detection, Detections
from klygo.utils.progress import ProgressBar


def _parse_data_yaml(yaml_path: str):
    """
    Hỗ trợ đọc nhanh file data.yaml (chuẩn YOLO dataset) để lấy thư mục ảnh test và danh sách nhãn.
    """
    import yaml
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Không tìm thấy file data: '{yaml_path}'")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = []
    if isinstance(config, dict):
        names_data = config.get("names", [])
        if isinstance(names_data, list):
            names = [str(n) for n in names_data]
        elif isinstance(names_data, dict):
            names = [str(v) for k, v in sorted(names_data.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))]

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    dataset_root = config.get("path", "") if isinstance(config, dict) else ""
    if not dataset_root:
        dataset_root = yaml_dir
    elif not os.path.isabs(str(dataset_root)):
        dataset_root = os.path.normpath(os.path.join(yaml_dir, str(dataset_root)))

    img_sub = (config.get("val") or config.get("train") or config.get("test") or "images") if isinstance(config, dict) else "images"
    if os.path.isabs(str(img_sub)):
        img_dir = str(img_sub)
    else:
        img_dir = os.path.normpath(os.path.join(str(dataset_root), str(img_sub)))

    return img_dir, names


class Detector(ABC):
    """
    Lớp cơ sở định nghĩa giao diện chung cho tất cả các mô hình nhận diện đối tượng trong Klygo.
    Tự động đảm nhiệm toàn bộ vòng lặp batching, tiến trình ProgressBar, quản lý bộ nhớ và thiết bị.
    """

    def __init__(self, metadata: Union[Dict[str, Any], str], **kwargs) -> None:
        if isinstance(metadata, str):
            self.metadata: Dict[str, Any] = {
                "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                "task": "Object-Detection",
                "backend": "Custom",
                "num_params": "Custom",
                "model_id": metadata,
            }
        else:
            self.metadata = dict(metadata)
            if "class" not in self.metadata:
                self.metadata["class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        self.class_name = self.metadata.get("class", self.__class__.__name__)
        self.task = self.metadata.get("task", "Object-Detection")
        self.backend = self.metadata.get("backend", "Custom")
        self.num_params = self.metadata.get("num_params", "Unknown")
        self.model_id = self.metadata.get("model_id", "")

        self._device: str = "cpu"
        self.half_mode: bool = False
        self.model: Any = None
        self.processor: Any = None

    @property
    def device(self) -> str:
        """Trả về tên thiết bị tính toán hiện tại của mô hình (ví dụ: 'cpu', 'cuda:0')."""
        if hasattr(self, "model") and hasattr(self.model, "device"):
            return str(self.model.device)
        return self._device

    def to(self, device_name: str) -> "Detector":
        """
        Tác dụng:
        - Chuyển toàn bộ trọng số mô hình lên thiết bị tính toán được chỉ định ('cpu', 'cuda', 'cuda:0'...).
        """
        self._device = device_name
        if hasattr(self, "model") and hasattr(self.model, "to"):
            try:
                import torch
                self.model.to(device_name)
                if device_name == "cpu" and hasattr(self.model, "parameters"):
                    if next(self.model.parameters()).dtype == torch.float16:
                        self.model = self.model.float()
                        self.half_mode = False
                elif "cuda" in device_name and self.half_mode:
                    self.model = self.model.half()
            except Exception:
                if hasattr(self.model, "device"):
                    self._device = str(self.model.device)
        return self

    def cpu(self) -> "Detector":
        """
        Tác dụng:
        - Chuyển mô hình về CPU và đưa về độ chính xác chuẩn FP32.
        """
        self.to("cpu")
        self.float()
        return self

    def half(self) -> "Detector":
        """
        Tác dụng:
        - Chuyển mô hình sang nửa độ chính xác FP16 (Half-Precision) để giảm 50% VRAM và tăng tốc GPU.
        """
        self.half_mode = True
        if hasattr(self, "model") and hasattr(self.model, "half"):
            if "cuda" in str(self.device):
                self.model = self.model.half()
        return self

    def float(self) -> "Detector":
        """
        Tác dụng:
        - Chuyển mô hình về độ chính xác chuẩn FP32 (Single-Precision).
        """
        self.half_mode = False
        if hasattr(self, "model") and hasattr(self.model, "float"):
            self.model = self.model.float()
        return self

    def warmup(self) -> None:
        """Khởi động mô hình với dữ liệu giả lập để nạp sẵn đồ thị tính toán lên GPU."""
        dummy_img = PIL.Image.new("RGB", (1, 1), color="black")
        try:
            self.predict(source=dummy_img, prompt=["dummy"], conf=0.9, text_threshold=0.9, verbose=False)
        except Exception:
            pass

    def unload(self) -> None:
        """Giải phóng mô hình khỏi GPU và dọn sạch bộ nhớ đệm VRAM."""
        self.cpu()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def export(self, output_dir: str) -> str:
        """
        Tác dụng:
        - Xuất toàn bộ mô hình (Weights + klygo.json) thành 1 thư mục độc lập để chạy Offline hoàn toàn.

        Đầu vào:
        - output_dir [str]: Đường dẫn thư mục cần xuất mô hình.

        Đầu ra:
        - [str]: Đường dẫn tuyệt đối tới thư mục mô hình đã xuất.
        """
        import json
        abs_out = os.path.abspath(output_dir)
        os.makedirs(abs_out, exist_ok=True)

        # 1. Lưu trọng số nếu mô hình hỗ trợ save_pretrained
        if hasattr(self, "processor") and hasattr(self.processor, "save_pretrained"):
            try:
                self.processor.save_pretrained(abs_out)
            except Exception:
                pass

        if hasattr(self, "model") and hasattr(self.model, "save_pretrained"):
            try:
                self.model.save_pretrained(abs_out)
            except Exception:
                pass

        # 2. Tạo và lưu file định danh klygo.json
        meta_to_save = dict(getattr(self, "metadata", {}))
        meta_to_save["class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        meta_to_save["model_id"] = abs_out
        klygo_json_path = os.path.join(abs_out, "klygo.json")
        with open(klygo_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2, ensure_ascii=False)

        return abs_out

    def save(self, output_dir: str) -> str:
        """Alias của phương thức export."""
        return self.export(output_dir)

    @abstractmethod
    def _infer_batch(
        self,
        batch_imgs: List[PIL.Image.Image],
        prompt: List[str],
        conf: float,
        text_threshold: float,
        half: bool,
    ) -> List[Detection]:
        """
        Phương thức trừu tượng cốt lõi: Thực thi suy luận cho 1 lô ảnh cụ thể.
        Các lớp mô hình con chỉ cần cài đặt duy nhất phương thức này.
        """
        raise NotImplementedError

    def predict(
        self,
        source: Any,
        prompt: Optional[Union[str, List[str]]] = None,
        conf: float = 0.25,
        text_threshold: float = 0.3,
        batch: int = 1,
        vid_stride: int = 1,
        max_frames: Optional[int] = None,
        half: bool = False,
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Detections:
        """
        Thực thi nhận diện đối tượng trên ảnh, video hoặc folder (luôn trả về tập hợp kết quả Detections).

        Đầu ra:
        - [Detections]: Tập hợp kết quả nhận diện (results[0] là kết quả Detection của ảnh đầu tiên).
        """
        actual_prompt = prompt or kwargs.get("classes") or kwargs.get("text_prompt")
        if actual_prompt is None:
            raise ValueError("Vui lòng cung cấp nhãn cần nhận diện qua 'prompt' hoặc 'classes'.")

        actual_conf = kwargs.get("threshold", conf)
        actual_batch = max(1, int(kwargs.get("batch_size", batch)))
        actual_stride = kwargs.get("step", kwargs.get("stride", vid_stride))

        if device is not None:
            self.to(device)
        if half:
            self.half()

        images, is_single = utils.resolve_images(source, step=actual_stride, max_frames=max_frames)
        if not images:
            return Detections(frames=[], source_type="list", fps=30.0)

        target_prompt = utils.normalize_prompt(actual_prompt)

        # 1. Trường hợp 1 ảnh đơn lẻ -> Vẫn trả về Detections chứa 1 frame
        if is_single:
            det = self._infer_batch(images, target_prompt, actual_conf, text_threshold, half)[0]
            det.image_frame_index = 0
            return Detections(frames=[det], source_type="image", fps=30.0)

        # 2. Trường hợp Video / Folder / Batch ảnh
        frame_results = []
        with ProgressBar(total=len(images), desc="Predict", unit="frame", verbose=verbose, colour="cyan") as pbar:
            for i in range(0, len(images), actual_batch):
                batch_imgs = images[i : i + actual_batch]
                dets = self._infer_batch(batch_imgs, target_prompt, actual_conf, text_threshold, half)
                for idx, det in enumerate(dets):
                    det.image_frame_index = i + idx
                frame_results.extend(dets)
                pbar.update(len(batch_imgs))

        return Detections(frames=frame_results, source_type="video" if len(images) > 1 else "image", fps=30.0)

    @abstractmethod
    def help(self) -> None:
        """In ra tài liệu hướng dẫn và danh sách các hàm nghiệp vụ của mô hình."""
        raise NotImplementedError

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
        """
        import time
        import torch

        if data:
            yaml_img_dir, yaml_names = _parse_data_yaml(data)
            source = source or yaml_img_dir
            text_prompt = text_prompt or yaml_names

        img = source if source is not None else PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        prompts = text_prompt or ["object"]

        # 1. Warmup
        for _ in range(warmup):
            self.predict(source=img, prompt=prompts, conf=threshold, verbose=False)

        # 2. Benchmark
        latencies = []
        is_cuda = "cuda" in str(getattr(self, "device", "cpu"))

        for _ in range(iterations):
            t_start = time.perf_counter()
            self.predict(source=img, prompt=prompts, conf=threshold, verbose=False)
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        min_latency = min(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

        w_dim, h_dim = (img.width, img.height) if isinstance(img, PIL.Image.Image) else (640, 640)

        report = {
            "model_id": getattr(self, "model_id", getattr(self, "name", self.__class__.__name__)),
            "backend": getattr(self, "backend", "Custom"),
            "device": getattr(self, "device", "cpu"),
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


# Alias tương thích ngược
DetectorModel = Detector
