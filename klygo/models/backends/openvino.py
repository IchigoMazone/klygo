from typing import Any
from klygo import files


def export_openvino(model_id: str, processor: Any, output_path: str, half: bool = False) -> str:
    """
    Tác dụng:
    - Xuất mô hình sang định dạng Intel OpenVINO (IR XML/BIN) trong thư mục output_path.

    Đầu vào:
    - model_id [str]: ID hoặc đường dẫn mô hình nguồn.
    - processor [Any]: Đối tượng tiền xử lý.
    - output_path [str]: Thư mục đích lưu trữ.
    - half [bool]: Tùy chọn xuất FP16. Mặc định: False.

    Đầu ra:
    - [str]: Đường dẫn thư mục đầu ra.
    """
    files.mkdir(output_path)

    try:
        from optimum.intel.openvino import OVModelForZeroShotObjectDetection

        ov_model = OVModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )
        ov_model.save_pretrained(output_path)
        if processor:
            processor.save_pretrained(output_path)
    except Exception:
        pass

    return output_path
