from typing import Any
from klygo import files


def export_openvino(
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
    - Xuất mô hình sang định dạng Intel OpenVINO (IR XML/BIN) trong thư mục output_path.

    Đầu vào:
    - model_id [str]: ID hoặc đường dẫn mô hình nguồn.
    - processor [Any]: Đối tượng tiền xử lý.
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
        from optimum.intel.openvino import OVModelForZeroShotObjectDetection

        ov_model = OVModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )
        if int8:
            try:
                from optimum.intel.openvino.quantization import OVQuantizer
                quantizer = OVQuantizer.from_pretrained(ov_model)
                quantizer.quantize(save_directory=output_path)
            except Exception:
                ov_model.save_pretrained(output_path)
        else:
            if half:
                try:
                    ov_model.half()
                except Exception:
                    pass
            ov_model.save_pretrained(output_path)

        if processor:
            processor.save_pretrained(output_path)
            if hasattr(processor, "image_processor"):
                processor.image_processor.save_pretrained(output_path)
    except ImportError:
        raise ImportError(
            "Để xuất định dạng OpenVINO, vui lòng cài đặt: pip install optimum[openvino] openvino nncf"
        )

    return output_path
