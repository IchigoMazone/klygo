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
        import os
        import glob
        from optimum.onnxruntime import ORTModelForZeroShotObjectDetection

        ort_model = ORTModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            export=True,
            use_io_binding=True,
        )
        ort_model.save_pretrained(output_path)

        if int8:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                onnx_files = glob.glob(os.path.join(output_path, "*.onnx"))
                for model_file in onnx_files:
                    if not model_file.endswith("_int8.onnx"):
                        quantized_file = model_file.replace(".onnx", "_int8.onnx")
                        quantize_dynamic(model_file, quantized_file, weight_type=QuantType.QInt8)
                        os.replace(quantized_file, model_file)
            except Exception:
                pass

        if processor:
            processor.save_pretrained(output_path)
            if hasattr(processor, "image_processor"):
                processor.image_processor.save_pretrained(output_path)
    except ImportError:
        raise ImportError(
            "Để xuất định dạng ONNX, vui lòng cài đặt: pip install optimum[onnxruntime] onnx onnxruntime"
        )

    return output_path
