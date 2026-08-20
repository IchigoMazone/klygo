from typing import Any
from klygo import files


def export_onnx(
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
    - Xuất mô hình sang định dạng ONNX trong thư mục output_path.

    Đầu vào:
    - model_id [str]: ID hoặc đường dẫn mô hình nguồn.
    - processor [Any]: Đối tượng tiền xử lý (Tokenizer/FeatureExtractor).
    - output_path [str]: Thư mục đích lưu trữ.
    - half [bool]: Tùy chọn xuất FP16. Mặc định: False.
    - int8 [bool]: Tùy chọn lượng tử hóa 8-bit INT8. Mặc định: False.
    - calibration_source [Any]: Nguồn ảnh hiệu chuẩn INT8.
    - calibration_prompts [Any]: Nhãn lớp hiệu chuẩn INT8.

    Đầu ra:
    - [str]: Đường dẫn thư mục đầu ra.
    """
    files.mkdir(output_path)

    try:
        from optimum.onnxruntime import ORTModelForZeroShotObjectDetection

        ort_model = ORTModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            export=True,
            use_io_binding=True,
        )
        ort_model.save_pretrained(output_path)
        if processor:
            processor.save_pretrained(output_path)
            if hasattr(processor, "image_processor"):
                processor.image_processor.save_pretrained(output_path)
    except ImportError:
        raise ImportError(
            "Để xuất định dạng ONNX, vui lòng cài đặt: pip install optimum[onnxruntime] onnx onnxruntime"
        )

    return output_path
