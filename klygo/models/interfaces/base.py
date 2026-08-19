from abc import ABC, abstractmethod
from typing import Any, List, Union
from .outputs import DetectionResult, CropResult


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
        text_prompt: List[str],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Tác dụng:
        - Thực thi suy luận nhận diện đối tượng không giới hạn tập nhãn (Zero-shot Detection) trên 1 ảnh duy nhất.

        Đầu vào:
        - source [Any]: 1 bức ảnh đầu vào (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần tìm kiếm.
        - threshold [float]: Ngưỡng lọc khung giới hạn (Confidence Threshold). Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản (Text Similarity Threshold). Mặc định: 0.3.

        Đầu ra:
        - [DetectionResult]: Đối tượng kết quả nhận diện chuẩn hóa.
        """
        raise NotImplementedError

    @abstractmethod
    def crop(
        self,
        source: Any,
        text_prompt: List[str],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> CropResult:
        """
        Tác dụng:
        - Nhận diện và cắt các đối tượng tìm thấy trên 1 ảnh duy nhất thành các ảnh con độc lập.

        Đầu vào:
        - source [Any]: 1 bức ảnh đầu vào (Đường dẫn file, PIL.Image, NumPy array hoặc PyTorch Tensor).
        - text_prompt [List[str]]: Danh sách các tên nhãn từ khóa cần cắt.
        - threshold [float]: Ngưỡng lọc khung giới hạn. Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản. Mặc định: 0.3.

        Đầu ra:
        - [CropResult]: Đối tượng tập hợp chứa các ảnh con và siêu dữ liệu tọa độ gốc.
        """
        raise NotImplementedError

    @abstractmethod
    def export(self, output_path: str, format: str = "onnx", half: bool = False) -> str:
        """
        Tác dụng:
        - Biên dịch và xuất mô hình sang các định dạng tối ưu hóa phần cứng (ONNX, TensorRT, OpenVINO, FP16).

        Đầu vào:
        - output_path [str]: Đường dẫn thư mục đích để lưu mô hình và cấu hình config.json.
        - format [str]: Định dạng xuất ('onnx', 'engine', 'openvino', 'torchscript', 'safetensors'). Mặc định: 'onnx'.
        - half [bool]: Tùy chọn sử dụng nửa độ chính xác FP16. Mặc định: False.

        Đầu ra:
        - [str]: Trả về đường dẫn tuyệt đối của thư mục đã xuất.
        """
        raise NotImplementedError

    @abstractmethod
    def dataset(
        self,
        output_path: str,
        format: str,
        source: Union[str, List[Any]],
        text_prompt: List[str],
        batch_size: int = 16,
        threshold: float = 0.4,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Tác dụng:
        - Tự động tạo bộ dữ liệu huấn luyện định dạng YOLO hoặc Classification từ nguồn ảnh/video.

        Đầu vào:
        - output_path [str]: Thư mục lưu trữ bộ dữ liệu đầu ra.
        - format [str]: Định dạng xuất ('yolo' cho Object Detection, 'classification' cho Image Classification).
        - source [str | List]: Đường dẫn thư mục ảnh, file video, hoặc danh sách ảnh đã đọc sẵn từ media.load.
        - text_prompt [List[str]]: Danh sách các lớp nhãn đối tượng cần trích xuất.
        - batch_size [int]: Kích thước xử lý theo lô. Mặc định: 16.
        - threshold [float]: Ngưỡng độ tin cậy nhận diện. Mặc định: 0.4.
        - verbose [bool]: Hiển thị thanh tiến trình ProgressBar. Mặc định: True.
        """
        raise NotImplementedError

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

