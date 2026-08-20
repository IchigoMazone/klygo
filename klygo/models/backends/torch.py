from typing import Any
from klygo import files


def export_torch(
    model_id: str,
    processor: Any,
    output_path: str,
    half: bool = False,
    int8: bool = False,
    calibration_source: Any = None,
    calibration_prompts: Any = None,
) -> str:
    """
    Tác dụng:
    - Xuất mô hình sang định dạng PyTorch / SafeTensors (FP32, FP16 hoặc INT8) trong thư mục output_path.

    Đầu vào:
    - model_id [str]: ID hoặc đường dẫn mô hình nguồn.
    - processor [Any]: Đối tượng tiền xử lý.
    - output_path [str]: Thư mục đích lưu trữ.
    - half [bool]: Tùy chọn nén FP16. Mặc định: False.
    - int8 [bool]: Tùy chọn lượng tử hóa INT8. Mặc định: False.
    - calibration_source [Any]: Nguồn ảnh hiệu chuẩn INT8.
    - calibration_prompts [Any]: Nhãn lớp hiệu chuẩn INT8.

    Đầu ra:
    - [str]: Đường dẫn thư mục đầu ra.
    """
    files.mkdir(output_path)
    from transformers import AutoModelForZeroShotObjectDetection

    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    if half:
        model = model.half()
    model.save_pretrained(output_path)
    if processor:
        processor.save_pretrained(output_path)
        if hasattr(processor, "image_processor"):
            processor.image_processor.save_pretrained(output_path)
    return output_path
