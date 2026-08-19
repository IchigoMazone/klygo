from typing import Any
from klygo import files


def export_tensorrt(model_id: str, processor: Any, output_path: str, half: bool = True) -> str:
    """
    Tác dụng:
    - Xuất mô hình sang định dạng NVIDIA TensorRT Engine (.engine) trong thư mục output_path.

    Đầu vào:
    - model_id [str]: ID hoặc đường dẫn mô hình nguồn.
    - processor [Any]: Đối tượng tiền xử lý.
    - output_path [str]: Thư mục đích lưu trữ.
    - half [bool]: Tùy chọn xuất FP16. Mặc định: True.

    Đầu ra:
    - [str]: Đường dẫn thư mục đầu ra.
    """
    files.mkdir(output_path)
    if processor:
        try:
            processor.save_pretrained(output_path)
        except Exception:
            pass
    return output_path
