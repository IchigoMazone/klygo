# 🏛️ HƯỚNG DẪN TÍCH HỢP MÔ HÌNH MỚI VÀO HỆ SINH THÁI KLYGO (V2)

Tài liệu này cung cấp **kiến trúc chuẩn hóa, nguyên tắc thiết kế và khung mã nguồn mẫu (Blueprint Template)** để phát triển và tích hợp bất kỳ mô hình AI nào (Zero-Shot Detection, YOLO, Florence-2, SAM, RT-DETR, Qwen-VL,...) vào hệ sinh thái `klygo.models`.

---

## 📑 MỤC LỤC
1. [Triết lý Thiết kế Cốt lõi của Klygo](#1-triết-lý-thiết-kế-cốt-lõi-của-klygo)
2. [Cấu trúc Giao diện Chuẩn (7 Phương Thức Nghiệp Vụ)](#2-cấu-trúc-giao-diện-chuẩn-7-phương-thức-nghiệp-vụ)
3. [Quy Chuẩn Đối Tượng Đầu Ra (Outputs)](#3-quy-chuẩn-đối-tượng-đầu-ra-outputs)
4. [Mẫu Khung Code Hoàn Chỉnh (Production-Ready Template)](#4-mẫu-khung-code-hoàn-chỉnh-production-ready-template)
5. [Đăng Ký Vào Hệ Thống Registry `models.load()`](#5-đăng-ký-vào-hệ-thống-registry-modelsload)
6. [Các Quy Tắc Vàng Cần Tuân Thủ Nghiêm Ngặt](#6-các-quy-tắc-vàng-cần-tuân-thủ-nghiêm-ngặt)

---

## 1. Triết lý Thiết kế Cốt lõi của Klygo

Mọi mô hình trong Klygo phải đảm bảo:
* **Tương thích 100% với định dạng YOLO**: Mọi hàm đều phải chấp nhận tham số `data="data.yaml"` để tự động nạp nguồn ảnh và danh sách nhãn lớp `names`.
* **Đa nền tảng (Notebook & Desktop)**: 
  * Trên **Google Colab / Jupyter**: `.show()` chỉ in đúng **1 ảnh inline duy nhất**, không double-render, hỗ trợ co giãn `width=...`.
  * Trên **Desktop IDE (VSCode / PyCharm / Terminal)**: `.show()` tự động bật trình xem ảnh của hệ điều hành.
* **Tương thích Matplotlib**: Mọi ảnh con cắt được (`crops[i]`) phải truyền trực tiếp được vào `matplotlib.pyplot.imshow(crops[i])` mà không phát sinh `TypeError`.
* **Đồ họa Bounding Box chuyên nghiệp**: Áp dụng thuật toán **2-Pass Rendering** (vẽ khung trước, vẽ nhãn chữ lên trên cùng) và **Collision Avoidance** (tránh đè nhãn).

---

## 2. Cấu trúc Giao diện Chuẩn (7 Phương Thức Nghiệp Vụ)

Mọi lớp mô hình phải kế thừa từ `DetectorModel` trong `klygo.models.interfaces` và triển khai đầy đủ 7 phương thức sau:

```
                              ┌─────────────────────────────────────────┐
                              │          DetectorModel (Base)           │
                              └────────────────────┬────────────────────┘
                                                   │
         ┌──────────────────┬──────────────────────┼─────────────────────┬──────────────────┐
         │                  │                      │                     │                  │
 ┌───────▼────────┐ ┌───────▼────────┐     ┌───────▼────────┐    ┌───────▼────────┐ ┌───────▼────────┐
 │ predict()      │ │ crop()         │     │ preview()      │    │ dataset()      │ │ export()       │
 │ Nhận diện 1 ảnh│ │ Cắt đối tượng  │     │ Batch/Video    │    │ Tạo dataset    │ │ FP16/INT8/ONNX │
 └────────────────┘ └────────────────┘     └────────────────┘    └────────────────┘ └────────────────┘
         │                  │
 ┌───────▼────────┐ ┌───────▼────────┐
 │ benchmark()    │ │ warmup/unload  │
 │ Đo Latency/FPS │ │ Quản lý GPU/VRAM│
 └────────────────┘ └────────────────┘
```

| STT | Phương Thức | Đầu Vào Chính | Đầu Ra Trả Về | Mô Tả |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `predict()` | `source`, `text_prompt`, `data`, `threshold` | `DetectionResult` | Suy luận nhận diện trên 1 ảnh duy nhất. |
| **2** | `crop()` | `source`, `text_prompt`, `data`, `threshold` | `CropResult` | Cắt các vật thể thành danh sách ảnh con `CroppedObject`. |
| **3** | `preview()` | `source`, `text_prompt`, `output_path`, `show` | `PreviewResult` | Chạy hàng loạt trên thư mục ảnh / file video, xuất file đúng định dạng đầu vào. |
| **4** | `dataset()` | `output_path`, `format`, `source`, `data` | `str` (Thư mục) | Auto-labeling tạo bộ dữ liệu train Detection hoặc Classification. |
| **5** | `export()` | `output_path`, `format`, `half`, `int8`, `data` | `str` (Thư mục) | Xuất mô hình sang SafeTensors, ONNX, TensorRT, OpenVINO (FP16/INT8). |
| **6** | `benchmark()` | `data`, `source`, `iterations`, `warmup` | `dict` (Báo cáo) | Đo đạc Latency (ms) và tốc độ FPS. |
| **7** | `to() / warmup() / unload() / help()` | `device_name` | `self` / `None` | Quản lý thiết bị tính toán và bộ nhớ VRAM. |

---

## 3. Quy Chuẩn Đối Tượng Đầu Ra (Outputs)

Mô hình **BẮT BUỘC** trả về các đối tượng chuẩn hóa có sẵn trong `klygo.models.interfaces`:

### 🔹 `DetectionResult`
* Thuộc tính: `image` (PIL.Image gốc), `objects` (List[DetectedObject]), `speed` (`{'inference': float}`).
* Phương thức:
  * `.show(width=None)` ➔ Hiển thị ảnh kèm BBox (trả về `None`).
  * `.save(path, width=None)` ➔ Lưu ảnh đã vẽ BBox ra đĩa.
  * `.plot()` ➔ Trả về đối tượng `PIL.Image` đã vẽ BBox theo chuẩn YOLO.
  * `.to_dict()` ➔ Trích xuất nhãn, score và tọa độ `[xmin, ymin, xmax, ymax]`.

### 🔹 `CropResult`
* Thuộc tính: `crops` (List[CroppedObject]), `labels` (List[str]).
* Phương thức:
  * `crops[i]` ➔ Truy cập từng ảnh con. Tương thích trực tiếp với `plt.imshow(crops[i])`.
  * `crops[i].show(width=200)` ➔ Mở xem từng ảnh con.
  * `crops.save(output_dir)` ➔ Lưu toàn bộ ảnh con ra thư mục.

### 🔹 `PreviewResult`
* Thuộc tính: `results` (List[DetectionResult]), `source_type` (`"video"` hoặc `"directory"`), `output_path`.
* Phương thức:
  * `.show(width=None, limit=None)` ➔ Xem trước kết quả.
  * `.save(output_path, fps=None)` ➔ Lưu video `.mp4` hoặc folder ảnh tương ứng.

---

## 4. Mẫu Khung Code Hoàn Chỉnh (Production-Ready Template)

Dưới đây là khung code chuẩn để bạn sao chép và triển khai mô hình mới (ví dụ `YoloWorldDetect` hoặc `MyCustomDetector`):

```python
import os
import time
import torch
import PIL.Image
from typing import Any, List, Optional, Dict, Union
from pathlib import Path

from klygo import files, media
from klygo.utils.progress import ProgressBar
from klygo.models.interfaces import (
    DetectorModel,
    DetectedObject,
    DetectionResult,
    CroppedObject,
    CropResult,
    PreviewResult,
)
from klygo.models.interfaces.base import _parse_data_yaml
from klygo.models.interfaces.outputs import _is_notebook


class CustomDetectorModel(DetectorModel):
    """
    Trình bao bọc mô hình nhận diện chuẩn hóa theo kiến trúc Klygo V2.
    """

    def __init__(
        self,
        task: str = "Object-Detection",
        backend: str = "PyTorch",
        num_params: str = "Base",
        model_id: str = "my-custom-model",
        device_map: Optional[str] = None,
        source_model_id: Optional[str] = None,
        half: bool = False,
        int8: bool = False,
        **kwargs,  # Luôn nhận **kwargs để tương thích khi nạp từ klygo.json
    ) -> None:
        self.task = task
        self.backend = backend
        self.num_params = num_params
        self.model_id = model_id
        self.source_model_id = source_model_id
        self.half = half
        self.int8 = int8

        # Khởi tạo thiết bị tính toán
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if device_map:
            self._device = device_map

        # TODO: Khởi tạo Processor / Weights của mô hình tại đây
        self.model = None
        self.processor = None

    @property
    def device(self) -> str:
        return str(self._device)

    def to(self, device_name: str) -> "CustomDetectorModel":
        self._device = device_name
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(device_name)
        return self

    # =========================================================================
    # 1. PREDICT: Nhận diện 1 ảnh
    # =========================================================================
    def predict(
        self,
        source: Any,
        text_prompt: Optional[Union[str, List[str]]] = None,
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        data: Optional[str] = None,
    ) -> DetectionResult:
        # Bóc tách data.yaml nếu có
        if data:
            _, yaml_names = _parse_data_yaml(data)
            if text_prompt is None and yaml_names:
                text_prompt = yaml_names

        # Chuẩn hóa prompt thành List[str]
        if isinstance(text_prompt, str):
            target_prompt = [text_prompt]
        elif isinstance(text_prompt, (list, tuple)):
            target_prompt = list(text_prompt)
        else:
            target_prompt = ["object"]

        # Chuẩn hóa ảnh đầu vào thành PIL.Image RGB
        pil_img = media.load(source, verbose=False)[0] if isinstance(source, str) else media.to_pil(source)
        w_img, h_img = pil_img.size

        t0 = time.perf_counter()
        
        # TODO: Chạy suy luận thực tế qua mô hình của bạn
        detected_objects: List[DetectedObject] = []
        # Ví dụ mẫu:
        # detected_objects.append(
        #     DetectedObject(label="cat", score=0.95, box=[10, 20, 100, 150], img_size=(w_img, h_img))
        # )

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        return DetectionResult(
            image=pil_img,
            objects=detected_objects,
            speed={"inference": latency_ms},
        )

    # =========================================================================
    # 2. CROP: Nhận diện và cắt các đối tượng thành ảnh con
    # =========================================================================
    def crop(
        self,
        source: Any,
        text_prompt: Optional[Union[str, List[str]]] = None,
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        data: Optional[str] = None,
    ) -> CropResult:
        res = self.predict(
            source=source,
            text_prompt=text_prompt,
            threshold=threshold,
            text_threshold=text_threshold,
            data=data,
        )
        crops = []
        w_img, h_img = res.image.size
        for obj in res.objects:
            xmin, ymin, xmax, ymax = obj.box
            crop_img = res.image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
            crops.append(
                CroppedObject(
                    image=crop_img,
                    label=obj.label,
                    score=obj.score,
                    box=obj.box,
                    original_size=(w_img, h_img),
                )
            )
        return CropResult(crops)

    # =========================================================================
    # 3. PREVIEW: Trực quan hóa trên Folder ảnh / Video / media.load
    # =========================================================================
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
    ) -> PreviewResult:
        import cv2 as cv
        from klygo.datasets.detect import _resolve_source

        if data:
            d_source, d_names = _parse_data_yaml(data)
            source = source or d_source
            text_prompt = text_prompt or d_names

        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
        is_video = isinstance(source, (str, Path)) and str(source).lower().endswith(video_extensions)

        results = []
        annotated_frames = []

        if is_video:
            cap = cv.VideoCapture(str(source))
            target_fps = fps or float(cap.get(cv.CAP_PROP_FPS) or 30.0)
            try:
                frame_idx = 0
                while True:
                    if limit is not None and frame_idx >= limit:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pil_frame = PIL.Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    res = self.predict(pil_frame, text_prompt=text_prompt, threshold=threshold, text_threshold=text_threshold)
                    results.append(res)
                    annotated_frames.append(res.plot())
                    frame_idx += 1
            finally:
                cap.release()

            final_output = None
            if output_path:
                final_output = output_path if str(output_path).lower().endswith(video_extensions) else os.path.join(output_path, "preview.mp4")
                media.save_video(final_output, annotated_frames, fps=target_fps, verbose=verbose)

            if show and results:
                results[0].show(width=width)

            return PreviewResult(results=results, source_type="video", output_path=final_output, annotated_frames=annotated_frames, fps=target_fps)

        else:
            images = _resolve_source(source)
            if limit:
                images = images[:limit]

            for idx, img in enumerate(images, 1):
                res = self.predict(img, text_prompt=text_prompt, threshold=threshold, text_threshold=text_threshold)
                ann = res.plot()
                results.append(res)
                annotated_frames.append(ann)
                if output_path:
                    files.mkdir(output_path)
                    media.save(os.path.join(output_path, f"annotated_{idx:05d}.jpg"), ann, overwrite=True, verbose=False)
                if show:
                    res.show(width=width)

            return PreviewResult(results=results, source_type="directory", output_path=output_path, annotated_frames=annotated_frames)

    # =========================================================================
    # 4. DATASET: Auto-labeling tạo tập dữ liệu huấn luyện
    # =========================================================================
    def dataset(
        self,
        output_path: str,
        format: str = "detection",
        source: Optional[Union[str, List[Any]]] = None,
        text_prompt: Optional[List[str]] = None,
        data: Optional[str] = None,
        batch_size: int = 16,
        threshold: float = 0.4,
        verbose: bool = True,
        **kwargs,
    ) -> str:
        from klygo.datasets import detect
        if data:
            d_source, d_names = _parse_data_yaml(data)
            source = source or d_source
            text_prompt = text_prompt or d_names

        return detect.export(
            model=self,
            output_path=output_path,
            format=format,
            source=source,
            text_prompt=text_prompt,
            batch_size=batch_size,
            threshold=threshold,
            verbose=verbose,
            **kwargs,
        )

    # =========================================================================
    # 5. EXPORT: Xuất mô hình đa backend (SafeTensors, ONNX, TensorRT, OpenVINO)
    # =========================================================================
    def export(
        self,
        output_path: str,
        format: str = "safetensors",
        half: bool = False,
        int8: bool = False,
        data: Optional[str] = None,
        calibration_source: Optional[Union[str, List[Any]]] = None,
        calibration_prompts: Optional[List[str]] = None,
    ) -> str:
        from klygo.models import backends

        if data:
            d_source, d_names = _parse_data_yaml(data)
            calibration_source = calibration_source or d_source
            calibration_prompts = calibration_prompts or d_names

        files.mkdir(output_path)
        format_lower = format.lower()

        if format_lower in ["onnx"]:
            backends.export_onnx(self.model_id, self.processor, output_path, half=half, int8=int8, calibration_source=calibration_source, calibration_prompts=calibration_prompts)
        elif format_lower in ["tensorrt", "engine", "trt"]:
            backends.export_tensorrt(self.model_id, self.processor, output_path, half=half, int8=int8, calibration_source=calibration_source, calibration_prompts=calibration_prompts)
        elif format_lower in ["openvino", "ov", "xml"]:
            backends.export_openvino(self.model_id, self.processor, output_path, half=half, int8=int8, calibration_source=calibration_source, calibration_prompts=calibration_prompts)
        else:
            backends.export_torch(self.model_id, self.processor, output_path, half=half, int8=int8, calibration_source=calibration_source, calibration_prompts=calibration_prompts)

        # Lưu file klygo.json để models.load() có thể nạp lại tự động
        config_data = {
            "class": self.__class__.__name__,
            "task": self.task,
            "backend": format_lower,
            "num_params": self.num_params,
            "model_id": output_path,
            "source_model_id": self.model_id,
            "half": half,
            "int8": int8,
        }
        files.save(os.path.join(output_path, "klygo.json"), config_data, overwrite=True, verbose=False)
        return output_path

    # =========================================================================
    # 6. BENCHMARK: Đánh giá tốc độ & độ trễ
    # =========================================================================
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
        return super().benchmark(
            source=source,
            text_prompt=text_prompt,
            data=data,
            iterations=iterations,
            warmup=warmup,
            threshold=threshold,
            verbose=verbose,
        )

    # =========================================================================
    # 7. CÁC HÀM TIỆN ÍCH
    # =========================================================================
    def warmup(self) -> None:
        dummy = PIL.Image.new("RGB", (64, 64), color="black")
        try:
            self.predict(dummy, text_prompt=["dummy"], threshold=0.9)
        except Exception:
            pass

    def unload(self) -> None:
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def help(self) -> None:
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print("=" * 52)
        print("1. predict(source, text_prompt, threshold=0.4)")
        print("2. crop(source, text_prompt, threshold=0.4)")
        print("3. preview(source, text_prompt, output_path=None, show=True)")
        print("4. dataset(output_path, format='detection', source=None, data=None)")
        print("5. export(output_path, format='safetensors', half=False, int8=False)")
        print("6. benchmark(data='data.yaml', iterations=20, warmup=5)")
```

---

## 5. Đăng Ký Vào Hệ Thống Registry `models.load()`

Để người dùng có thể nạp mô hình qua lệnh `models.load("tên-mô-hình")` hoặc `models.load("./thư-mục-export")`, bạn chỉ cần thực hiện 2 bước đơn giản:

### Bước 1: Đăng ký lớp vào `klygo/models/load.py`
```python
# klygo/models/load.py
from .detection.custom_detector import CustomDetectorModel

CLASS_MAPPING = {
    "GroundingDinoDetect": GroundingDinoDetect,
    "CustomDetectorModel": CustomDetectorModel,  # <-- Thêm vào đây
}
```

### Bước 2: Thêm tên định danh vào `klygo/models/registry.json`
```json
{
  "my-custom-model": {
    "class": "CustomDetectorModel",
    "task": "Object-Detection",
    "backend": "Hugging Face",
    "num_params": "Tiny",
    "model_id": "my-org/my-custom-model-weights"
  }
}
```

---

## 6. Các Quy Tắc Vàng Cần Tuân Thủ Nghiêm Ngặt

1. **Không in đúp 2 ảnh trên Colab/Jupyter**:
   * Các hàm `.show()` trong `DetectionResult`, `CroppedObject`, `CropResult`, `PreviewResult` **PHẢI trả về `None`** (không được `return img`).
2. **Khắc phục lỗi Matplotlib `imshow`**:
   * Các đối tượng chứa ảnh con (như `CroppedObject`) phải luôn khai báo phương thức `__array__`:
     ```python
     def __array__(self, dtype=None):
         import numpy as np
         return np.asarray(self.image, dtype=dtype)
     ```
3. **Lọc kwargs an toàn khi nạp mô hình**:
   * Hàm khởi tạo `__init__` luôn phải nhận `int8: bool = False` và `**kwargs` để khi nạp lại từ các file `klygo.json` cũ hoặc mới đều không bị lỗi `unexpected keyword argument`.
4. **Trình diễn Bounding Box không che nhãn**:
   * Luôn sử dụng hàm `plot()` có sẵn trong `DetectionResult` vì đã tích hợp sẵn cơ chế **2-Pass Drawing** và **Collision Avoidance**.
