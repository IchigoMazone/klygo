from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional, Dict
from .outputs import DetectionResult, CropResult


def _parse_data_yaml(data_path: Any) -> tuple:
    """
    Trích xuất đường dẫn thư mục ảnh và danh sách nhãn lớp từ file data.yaml (chuẩn YOLO).
    """
    if not isinstance(data_path, str) or not data_path.endswith((".yaml", ".yml")):
        return None, []

    import os
    import yaml
    from klygo import files

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình data.yaml tại: {data_path}")

    try:
        config = files.load(data_path, verbose=False)
    except Exception:
        config = None

    if isinstance(config, str):
        try:
            config = yaml.safe_load(config)
        except Exception:
            pass

    if not isinstance(config, dict) or not config:
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read().strip().strip("'\"")
            config = yaml.safe_load(content)
        except Exception:
            config = None

    if not isinstance(config, dict) or not config:
        import re
        config = {}
        names_dict = {}
        in_names = False
        with open(data_path, "r", encoding="utf-8") as f:
            raw_lines = f.read().replace("\\n", "\n").splitlines()
        for line in raw_lines:
            line_str = line.strip().strip("'\"")
            if not line_str or line_str.startswith("#"):
                continue
            if line_str.startswith("path:"):
                config["path"] = line_str.split("path:", 1)[1].strip().strip("'\"")
            elif line_str.startswith("val:"):
                config["val"] = line_str.split("val:", 1)[1].strip().strip("'\"")
            elif line_str.startswith("train:"):
                config["train"] = line_str.split("train:", 1)[1].strip().strip("'\"")
            elif line_str.startswith("names:"):
                in_names = True
                val_part = line_str.split("names:", 1)[1].strip()
                if val_part.startswith("[") and val_part.endswith("]"):
                    config["names"] = [x.strip().strip("'\"") for x in val_part[1:-1].split(",") if x.strip()]
                    in_names = False
            elif in_names:
                m = re.match(r"^(\d+)\s*:\s*(.+)$", line_str)
                if m:
                    names_dict[int(m.group(1))] = m.group(2).strip().strip("'\"")
                elif line_str.startswith("- "):
                    if "names" not in config:
                        config["names"] = []
                    config["names"].append(line_str[2:].strip().strip("'\""))
                elif ":" in line_str:
                    in_names = False
        if names_dict:
            config["names"] = [names_dict[k] for k in sorted(names_dict.keys())]

    # 1. Trích xuất danh sách names:
    names_raw = config.get("names", []) if isinstance(config, dict) else []
    names = []
    if isinstance(names_raw, dict):
        for k in sorted(names_raw.keys()):
            names.append(str(names_raw[k]))
    elif isinstance(names_raw, list):
        names = [str(n) for n in names_raw]

    # 2. Trích xuất đường dẫn ảnh (ưu tiên val -> train -> images):
    dataset_root = config.get("path", "") if isinstance(config, dict) else ""
    yaml_dir = os.path.dirname(os.path.abspath(data_path))
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


class DetectorModel(ABC):
    """
    Lớp cơ sở trừu tượng thuần túy định nghĩa giao diện chung cho tất cả các mô hình nhận diện trong Klygo.
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Trả về tên thiết bị tính toán hiện tại của mô hình (ví dụ: 'cpu', 'cuda:0')."""
        raise NotImplementedError

    @abstractmethod
    def to(self, device_name: str) -> "DetectorModel":
        """
        Tác dụng:
        - Chuyển toàn bộ trọng số mô hình lên thiết bị tính toán được chỉ định.

        Đầu vào:
        - device_name [str]: Tên thiết bị ('cpu', 'cuda', 'cuda:0').

        Đầu ra:
        - [DetectorModel]: Trả về chính đối tượng mô hình (hỗ trợ method chaining).
        """
        raise NotImplementedError

    @abstractmethod
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
        - Thực thi suy luận nhận diện đối tượng không giới hạn tập nhãn (Zero-shot Detection) trên 1 ảnh duy nhất.

        Đầu vào:
        - source [Any]: 1 bức ảnh đầu vào (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần tìm kiếm (hoặc nạp qua data='data.yaml').
        - threshold [float]: Ngưỡng lọc khung giới hạn (Confidence Threshold). Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản (Text Similarity Threshold). Mặc định: 0.3.
        - data [str]: Đường dẫn file data.yaml (tự động lấy nhãn lớp giống YOLO).

        Đầu ra:
        - [DetectionResult]: Đối tượng kết quả nhận diện chuẩn hóa.
        """
        raise NotImplementedError

    @abstractmethod
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
        - Biên dịch và xuất mô hình sang các định dạng tối ưu hóa phần cứng (ONNX, TensorRT, OpenVINO, FP16, INT8).

        Đầu vào:
        - output_path [str]: Đường dẫn thư mục đích để lưu mô hình và cấu hình config.json.
        - format [str]: Định dạng xuất ('onnx', 'tensorrt', 'openvino', 'safetensors'). Mặc định: 'onnx'.
        - half [bool]: Tùy chọn sử dụng nửa độ chính xác FP16. Mặc định: False.
        - int8 [bool]: Tùy chọn lượng tử hóa 8-bit INT8. Mặc định: False.
        - data [str]: File cấu hình data.yaml để làm tập hiệu chuẩn INT8 Calibration.

        Đầu ra:
        - [str]: Trả về đường dẫn tuyệt đối của thư mục đã xuất.
        """
        raise NotImplementedError

    @abstractmethod
    def preview(
        self,
        source: Optional[Union[str, List[Any]]] = None,
        text_prompt: Optional[Union[str, List[str]]] = None,
        output_path: Optional[str] = None,
        show: bool = True,
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        data: Optional[str] = None,
        fps: Optional[float] = None,
        limit: Optional[int] = None,
        width: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """
        Tác dụng:
        - Chạy nhận diện trực quan hóa hàng loạt (Preview) trên thư mục ảnh, video hoặc danh sách ảnh từ media.load.
        - Tự động xuất ra thư mục ảnh hoặc file video theo đúng định dạng đầu vào.

        Đầu vào:
        - source [str | List[Any] | None]: Thư mục ảnh, file video (.mp4, .avi...), file ảnh hoặc danh sách ảnh từ media.load.
        - text_prompt [str | List[str] | None]: Danh sách từ khóa cần tìm kiếm (hoặc nạp qua data='data.yaml').
        - output_path [str | None]: Đường dẫn file video hoặc thư mục ảnh đầu ra cần lưu.
        - show [bool]: Hiển thị trực tiếp kết quả xem trước trên Notebook / Desktop. Mặc định: True.
        - threshold [float]: Ngưỡng lọc độ tin cậy. Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản. Mặc định: 0.3.
        - data [str | None]: Đường dẫn file data.yaml chuẩn YOLO (tự động bóc tách ảnh và nhãn).
        - fps [float | None]: Tốc độ khung hình (Frames Per Second) khi lưu video đầu ra.
        - limit [int | None]: Giới hạn số lượng ảnh / khung hình xử lý tối đa (tùy chọn).
        - width [int | None]: Chiều rộng hiển thị ảnh/video trên giao diện Colab/Jupyter.
        - verbose [bool]: Hiển thị thanh tiến trình xử lý ProgressBar. Mặc định: True.

        Đầu ra:
        - [PreviewResult]: Đối tượng tập hợp kết quả trực quan hóa, hỗ trợ .show(), .save().
        """
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
        import PIL.Image
        import torch

        if data:
            yaml_img_dir, yaml_names = _parse_data_yaml(data)
            source = source or yaml_img_dir
            text_prompt = text_prompt or yaml_names

        img = source if source is not None else PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        prompts = text_prompt or ["object"]

        # 1. Warmup
        for _ in range(warmup):
            self.predict(img, text_prompt=prompts, threshold=threshold)

        # 2. Benchmark
        latencies = []
        is_cuda = "cuda" in str(getattr(self, "device", "cpu"))

        for _ in range(iterations):
            t_start = time.perf_counter()
            self.predict(img, text_prompt=prompts, threshold=threshold)
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

    @abstractmethod
    def warmup(self) -> None:
        """Khởi động mô hình với dữ liệu giả lập để nạp sẵn đồ thị tính toán lên bộ nhớ."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Giải phóng mô hình khỏi GPU và dọn sạch bộ nhớ đệm VRAM."""
        raise NotImplementedError

    @abstractmethod
    def help(self) -> None:
        """In ra tài liệu hướng dẫn và danh sách các hàm nghiệp vụ của mô hình."""
        raise NotImplementedError

