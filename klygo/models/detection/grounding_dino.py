import os
import time
import logging
import warnings
import inspect
import torch
import PIL.Image
from typing import Any, List, Optional, Dict, Union

# Tắt các cảnh báo không cần thiết từ Hugging Face Hub và Transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=".*The key `labels` is will return integer ids.*")
warnings.filterwarnings("ignore", message=".*text_labels.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

try:
    import huggingface_hub.utils.logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

try:
    import transformers.utils.logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass

for _logger_name in ["huggingface_hub", "huggingface_hub.utils._http", "transformers"]:
    try:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
    except Exception:
        pass

from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo import files, media
from ..interfaces import (
    DetectorModel,
    CropResult,
    CropResults,
    DetectionResult,
    DetectionResults,
    DetectedObject,
    CroppedObject,
    PreviewResult,
)
from .. import backends


def _load_source_image(image: Any) -> PIL.Image.Image:
    """
    Chỉ chấp nhận duy nhất ảnh được nạp từ klygo.media.load (PIL.Image hoặc List[PIL.Image]).
    """
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    elif isinstance(image, (list, tuple)) and len(image) > 0 and isinstance(image[0], PIL.Image.Image):
        return image[0].convert("RGB")
    else:
        raise TypeError(
            f"Đầu vào 'image' không hợp lệ ({type(image).__name__}). "
            "model.predict() chỉ nhận duy nhất ảnh từ klygo.media.load() (PIL.Image hoặc List[PIL.Image]). "
            "Ví dụ: img = media.load('image.jpg'); model.predict(img, text_prompt=['cat'])"
        )


class GroundingDinoDetect(DetectorModel):
    """
    Trình bao bọc mô hình Zero-shot Object Detection kiến trúc Grounding DINO từ Hugging Face.
    Hỗ trợ nạp Online hoặc Offline trực tiếp từ thư mục trọng số cục bộ.
    """

    def __init__(
        self,
        task: str,
        backend: str,
        num_params: str,
        model_id: str,
        device_map: Optional[str] = None,
        source_model_id: Optional[str] = None,
        half: bool = False,
        int8: bool = False,
        **kwargs,
    ) -> None:
        self.task = task
        self.backend = backend
        self.num_params = num_params
        self.model_id = model_id
        self.source_model_id = source_model_id
        self.half = half
        self.int8 = int8

        load_path = model_id
        # Hỗ trợ nạp Offline khi model_id là thư mục cục bộ
        if files.is_dir(model_id) and source_model_id:
            if not files.exists(
                os.path.join(model_id, "model.safetensors")
            ) and not files.exists(os.path.join(model_id, "pytorch_model.bin")):
                load_path = source_model_id

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.processor = AutoProcessor.from_pretrained(load_path)
            if device_map:
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    load_path, device_map=device_map
                )
                self._device = str(self.model.device)
            else:
                self._device = "cpu"
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(load_path)
                if half:
                    self.model = self.model.half()
                elif self._device == "cpu":
                    try:
                        if next(self.model.parameters()).dtype == torch.float16:
                            self.model = self.model.float()
                    except Exception:
                        pass
                self.model.to(self._device)

        self.model.eval()

        # Cache trước chữ ký hàm để tăng tốc độ suy luận O(1) mà không tốn chi phí inspect
        sig = inspect.signature(self.processor.post_process_grounded_object_detection)
        self._has_input_ids = "input_ids" in sig.parameters
        self._has_threshold = "threshold" in sig.parameters
        self._has_box_threshold = "box_threshold" in sig.parameters
        self._has_text_threshold = "text_threshold" in sig.parameters

    @property
    def device(self) -> str:
        """Trả về tên thiết bị tính toán hiện tại của mô hình."""
        if hasattr(self.model, "device"):
            return str(self.model.device)
        return self._device

    def to(self, device_name: str) -> "GroundingDinoDetect":
        """
        Tác dụng:
        - Chuyển toàn bộ trọng số mô hình lên thiết bị tính toán được chỉ định ('cpu', 'cuda').
        """
        try:
            self.model.to(device_name)
            self._device = device_name
            # Nếu chuyển về CPU mà mô hình đang là float16, tự động chuyển về float32 để CPU tính toán được
            if device_name == "cpu" and next(self.model.parameters()).dtype == torch.float16:
                self.model = self.model.float()
            elif "cuda" in device_name and self.half:
                self.model = self.model.half()
        except ValueError:
            self._device = str(self.model.device)
        return self

    def predict(
        self,
        image: Any,
        text_prompt: Union[str, List[str]],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Tác dụng:
        - Nhận diện đối tượng trên 1 bức ảnh duy nhất (PIL.Image hoặc kết quả media.load).

        Đầu vào:
        - image [Any]: 1 bức ảnh đầu vào (kết quả từ media.load hoặc PIL.Image).
        - text_prompt [str | List[str]]: Danh sách các tên nhãn từ khóa cần tìm kiếm.
        - threshold [float]: Ngưỡng lọc khung giới hạn (Confidence Threshold). Mặc định: 0.4.
        - text_threshold [float]: Ngưỡng tương đồng văn bản (Text Similarity Threshold). Mặc định: 0.3.

        Đầu ra:
        - [DetectionResult]: Đối tượng kết quả chứa danh sách tọa độ, nhãn, độ tin cậy và tốc độ suy luận.
        """
        import time

        if isinstance(text_prompt, str):
            target_prompt = [text_prompt]
        elif isinstance(text_prompt, (list, tuple)):
            target_prompt = list(text_prompt)
        else:
            raise TypeError("text_prompt phải là chuỗi ký tự (str) hoặc danh sách chuỗi ký tự (list[str]).")

        t0 = time.perf_counter()
        pil_image = _load_source_image(image)

        text_labels = [target_prompt]
        inputs = self.processor(images=pil_image, text=text_labels, return_tensors="pt")

        # Xác định thiết bị tính toán
        model_device = getattr(self.model, "device", self._device)
        is_cuda = "cuda" in str(model_device)

        # Trên CPU: PyTorch MSDeformAttn grid_sample không hỗ trợ float16, tự động chuyển model sang float32
        if not is_cuda and next(self.model.parameters()).dtype == torch.float16:
            self.model = self.model.float()

        model_dtype = next(self.model.parameters()).dtype
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(device=model_device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(device=model_device)

        t1 = time.perf_counter()

        with torch.no_grad():
            if is_cuda and self.half:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        if is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        t2 = time.perf_counter()

        target_sizes = [pil_image.size[::-1]]
        post_kwargs = {
            "outputs": outputs,
            "target_sizes": target_sizes,
        }

        if self._has_input_ids:
            post_kwargs["input_ids"] = inputs.input_ids

        if self._has_threshold:
            post_kwargs["threshold"] = threshold
        else:
            if self._has_box_threshold:
                post_kwargs["box_threshold"] = threshold
            if self._has_text_threshold:
                post_kwargs["text_threshold"] = text_threshold

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            results = self.processor.post_process_grounded_object_detection(**post_kwargs)

        result = results[0]
        detected_objects = []
        labels_list = result.get("text_labels", result.get("labels", []))
        for idx, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], labels_list)):
            coords = box.tolist()
            detected_objects.append(
                CropResult(
                    id=idx,
                    label=str(label),
                    score=round(score.item(), 3),
                    box=[round(x, 2) for x in coords],
                    parent_image=pil_image,
                )
            )

        t3 = time.perf_counter()

        speed = {
            "preprocess": round((t1 - t0) * 1000, 2),
            "inference": round((t2 - t1) * 1000, 2),
            "postprocess": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        }

        return DetectionResult(
            source_image=pil_image,
            objects=detected_objects,
            speed=speed,
            image_frame_index=0,
            text_prompt=text_prompt,
            threshold=threshold,
            text_threshold=text_threshold,
        )

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
        - Xuất mô hình sang các kiến trúc tối ưu (ONNX, TensorRT, OpenVINO, FP16, INT8) kèm config.json.
        """
        from ..interfaces.base import _parse_data_yaml

        if data:
            d_source, d_names = _parse_data_yaml(data)
            calibration_source = calibration_source or d_source
            calibration_prompts = calibration_prompts or d_names

        output_path = os.path.abspath(output_path)
        files.mkdir(output_path)
        format = format.lower().lstrip(".")

        if format == "onnx":
            backends.export_onnx(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )
        elif format in ["engine", "tensorrt", "trt"]:
            backends.export_tensorrt(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )
        elif format in ["openvino", "xml", "ov"]:
            backends.export_openvino(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )
        elif format in ["torchscript", "pt", "safetensors", "torch", "fp16"]:
            backends.export_torch(
                self.model_id,
                self.processor,
                output_path,
                half=half,
                int8=int8,
                calibration_source=calibration_source,
                calibration_prompts=calibration_prompts,
            )

        config_data = {
            "class": "GroundingDinoDetect",
            "task": self.task,
            "backend": format,
            "num_params": self.num_params,
            "model_id": output_path,
            "source_model_id": self.model_id,
            "half": half,
            "int8": int8,
        }
        klygo_file = os.path.join(output_path, "klygo.json")
        files.save(klygo_file, config_data, overwrite=True, verbose=False)

        return output_path

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
        from ..interfaces.base import _parse_data_yaml

        if data:
            d_source, d_names = _parse_data_yaml(data)
            source = source or d_source
            text_prompt = text_prompt or d_names

        img = source if source is not None else PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        prompts = text_prompt or ["object"]

        # 1. Warmup
        for _ in range(warmup):
            self.predict(img, text_prompt=prompts, threshold=threshold)

        # 2. Benchmark
        latencies = []
        is_cuda = "cuda" in str(getattr(self.model, "device", self._device))

        for _ in range(iterations):
            t_start = time.perf_counter()
            self.predict(img, text_prompt=prompts, threshold=threshold)
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

        w_dim, h_dim = (img.width, img.height) if isinstance(img, PIL.Image.Image) else (640, 640)

        report = {
            "model_id": self.model_id,
            "backend": self.backend,
            "device": self.device,
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

    def warmup(self) -> None:
        """Khởi động mô hình với dữ liệu giả lập."""
        dummy_img = PIL.Image.new("RGB", (1, 1), color="black")
        try:
            self.predict(image=dummy_img, text_prompt=["dummy"], threshold=0.9, text_threshold=0.9)
        except Exception:
            pass

    def unload(self) -> None:
        """Giải phóng mô hình khỏi GPU và dọn sạch bộ nhớ đệm VRAM."""
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inference(
        self,
        images: Any,
        text_prompt: Union[str, List[str]],
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> DetectionResults:
        """
        Tác dụng:
        - Thực thi suy luận trên toàn bộ Video hoặc Thư mục ảnh nạp từ klygo.media.load.

        Đầu vào:
        - images: Dữ liệu video / folder nạp từ klygo.media.load (chỉ nhận List[PIL.Image] hoặc Generator).
        - text_prompt: Danh sách tên nhãn từ khóa cần tìm kiếm.
        - threshold: Ngưỡng lọc khung giới hạn (Confidence Threshold). Mặc định: 0.4.
        - text_threshold: Ngưỡng tương đồng văn bản (Text Similarity Threshold). Mặc định: 0.3.
        - batch_size: Số lượng ảnh xử lý đồng thời trong một batch (mặc định: 1).
        - verbose: Hiển thị thanh tiến trình ProgressBar khi suy luận. Mặc định: True.

        Đầu ra:
        - [DetectionResults]: Tập hợp kết quả nhận diện của toàn bộ video / folder.
        """
        if isinstance(images, (str, Path)):
            raise TypeError(
                "model.inference() chỉ nhận dữ liệu đã nạp từ klygo.media.load "
                "(ví dụ: images = media.load('video.mp4') hoặc images = media.load('folder_anh/')). "
                "Vui lòng sử dụng images = media.load(...) trước khi truyền vào."
            )

        # Chuyển đổi sang list ảnh
        if isinstance(images, PIL.Image.Image):
            image_list = [images]
        elif isinstance(images, (list, tuple)):
            image_list = list(images)
        elif hasattr(images, "__iter__"):
            image_list = list(images)
        else:
            raise TypeError(
                f"Định dạng đầu vào {type(images)} không hợp lệ. "
                "model.inference() chỉ nhận dữ liệu từ klygo.media.load (List[PIL.Image.Image] hoặc Generator)."
            )

        if not image_list:
            return DetectionResults(frames=[], source_type="list", fps=30.0)

        # Chuyển đổi np.ndarray sang PIL.Image nếu cần
        cleaned_images = []
        for img in image_list:
            if isinstance(img, PIL.Image.Image):
                cleaned_images.append(img)
            elif hasattr(img, "shape"):
                cleaned_images.append(PIL.Image.fromarray(img))
            else:
                raise TypeError(
                    f"Phần tử trong images có kiểu {type(img)} không phải ảnh từ media.load."
                )

        batch_size = max(1, int(batch_size))

        if isinstance(text_prompt, str):
            target_prompt = [text_prompt]
        elif isinstance(text_prompt, (list, tuple)):
            target_prompt = list(text_prompt)
        else:
            raise TypeError("text_prompt phải là chuỗi ký tự (str) hoặc danh sách chuỗi ký tự (list[str]).")

        model_device = getattr(self.model, "device", self._device)
        is_cuda = "cuda" in str(model_device)
        if not is_cuda and next(self.model.parameters()).dtype == torch.float16:
            self.model = self.model.float()
        model_dtype = next(self.model.parameters()).dtype

        from klygo.utils.progress import ProgressBar
        frame_results = []
        with ProgressBar(
            total=len(cleaned_images),
            desc="Inference",
            unit="frame",
            verbose=verbose,
            colour="cyan",
        ) as pbar:
            for b_idx in range(0, len(cleaned_images), batch_size):
                batch_imgs = cleaned_images[b_idx:b_idx + batch_size]
                global_indices = list(range(b_idx, b_idx + len(batch_imgs)))

                if len(batch_imgs) == 1:
                    res = self.predict(
                        image=batch_imgs[0],
                        text_prompt=text_prompt,
                        threshold=threshold,
                        text_threshold=text_threshold,
                    )
                    res.image_frame_index = global_indices[0]
                    frame_results.append(res)
                    pbar.update(1)
                    continue

                t0 = time.perf_counter()
                text_labels = [target_prompt] * len(batch_imgs)
                inputs = self.processor(images=batch_imgs, text=text_labels, return_tensors="pt")

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if torch.is_floating_point(v):
                            inputs[k] = v.to(device=model_device, dtype=model_dtype)
                        else:
                            inputs[k] = v.to(device=model_device)

                t1 = time.perf_counter()
                with torch.no_grad():
                    if is_cuda and self.half:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)

                if is_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                target_sizes = [img.size[::-1] for img in batch_imgs]
                post_kwargs = {
                    "outputs": outputs,
                    "target_sizes": target_sizes,
                }
                if self._has_input_ids:
                    post_kwargs["input_ids"] = inputs.input_ids
                if self._has_threshold:
                    post_kwargs["threshold"] = threshold
                else:
                    if self._has_box_threshold:
                        post_kwargs["box_threshold"] = threshold
                    if self._has_text_threshold:
                        post_kwargs["text_threshold"] = text_threshold

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    batch_outputs = self.processor.post_process_grounded_object_detection(**post_kwargs)

                t3 = time.perf_counter()
                batch_speed = {
                    "preprocess": round(((t1 - t0) * 1000) / len(batch_imgs), 2),
                    "inference": round(((t2 - t1) * 1000) / len(batch_imgs), 2),
                    "postprocess": round(((t3 - t2) * 1000) / len(batch_imgs), 2),
                    "total": round(((t3 - t0) * 1000) / len(batch_imgs), 2),
                }

                for out_res, pil_img, g_idx in zip(batch_outputs, batch_imgs, global_indices):
                    detected_objects = []
                    labels_list = out_res.get("text_labels", out_res.get("labels", []))
                    for idx, (box, score, label) in enumerate(zip(out_res["boxes"], out_res["scores"], labels_list)):
                        coords = box.tolist()
                        detected_objects.append(
                            CropResult(
                                id=idx,
                                label=str(label),
                                score=round(score.item(), 3),
                                box=[round(x, 2) for x in coords],
                                parent_image=pil_img,
                            )
                        )
                    frame_res = DetectionResult(
                        source_image=pil_img,
                        objects=detected_objects,
                        speed=batch_speed,
                        image_frame_index=g_idx,
                        text_prompt=text_prompt,
                        threshold=threshold,
                        text_threshold=text_threshold,
                    )
                    frame_results.append(frame_res)

                pbar.update(len(batch_imgs))

        source_type = "video" if len(cleaned_images) > 1 else "image"
        return DetectionResults(frames=frame_results, source_type=source_type, fps=30.0)

    def help(self) -> None:
        """In ra thông tin mô hình và danh sách các hàm nghiệp vụ."""
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print("=" * 52)
        print("1. predict(image, text_prompt, threshold=0.4, text_threshold=0.3)")
        print("   Nhan dien doi tuong tren 1 anh tu media.load (PIL.Image).")
        print("2. inference(images, text_prompt, threshold=0.4, text_threshold=0.3, batch_size=1)")
        print("   Suy luan tren toan bo Video hoac Folder anh tu media.load (ho tro batch_size).")
        print("3. export(output_path, format='safetensors', half=False, int8=False, data=None)")
        print("   Xuat mo hinh sang SafeTensors, ONNX, TensorRT, OpenVINO (FP16, INT8).")
        print("4. benchmark(data='data.yaml', iterations=20, warmup=5)")
        print("   Cham diem danh gia toc do suy luan (Do tre Latency ms / Toc do FPS).")
