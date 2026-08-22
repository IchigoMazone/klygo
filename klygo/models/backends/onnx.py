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

    # 1. Kiểm tra thư viện bắt buộc
    try:
        import onnx
        import onnxruntime
    except ImportError:
        raise ImportError(
            "Để xuất định dạng ONNX, vui lòng cài đặt: pip install optimum onnx onnxruntime"
        )

    import os
    import glob

    # 2. Thử các phương thức xuất ONNX phù hợp với kiến trúc mô hình
    exported = False
    export_error = None

    # Phương thức 1: Optimum main_export (Chuẩn nhất cho Hugging Face Transformers)
    try:
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=model_id,
            output=output_path,
            task="zero-shot-object-detection",
            trust_remote_code=True,
            device="cpu",
            fp16=half,
        )
        exported = True
    except Exception as e:
        export_error = e

    # Phương thức 2: Optimum ORTModelForCustomTasks
    if not exported:
        try:
            from optimum.onnxruntime import ORTModelForCustomTasks
            ort_model = ORTModelForCustomTasks.from_pretrained(
                model_id,
                export=True,
                use_io_binding=True,
            )
            ort_model.save_pretrained(output_path)
            exported = True
        except Exception as e:
            export_error = e

    # Phương thức 3: Báo lỗi chi tiết nếu không xuất được
    if not exported:
        raise RuntimeError(
            f"Không thể xuất mô hình '{model_id}' sang định dạng ONNX. "
            f"Chi tiết lỗi từ Optimum: {export_error}"
        )

    # 3. Lượng tử hóa INT8 Dynamic Quantization nếu yêu cầu
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

    # 4. Lưu processor kèm theo
    if processor:
        try:
            processor.save_pretrained(output_path)
            if hasattr(processor, "image_processor"):
                processor.image_processor.save_pretrained(output_path)
        except Exception:
            pass

    return output_path
