"""
Lớp cơ sở chuyên biệt cho bài toán Nhận diện Đối tượng (klygo.models.detection.base).
TẦNG 2: Task Engine Interface + Implementation - Cài đặt toàn bộ Động cơ Detection,
phân giải đa nguồn qua media.load, Batching loop, ProgressBar, đo thời gian ngoài luồng,
quản lý phần cứng, export và benchmark.
"""

import os
import time
from abc import abstractmethod
from typing import Dict, Any, Optional, Union, Sequence, Set, List
import PIL.Image

from klygo.models.base import BaseModel, override
from klygo.models import utils
from klygo.outputs.detect import Detections, Detection, Box
from klygo.utils.progress import ProgressBar


class Detector(BaseModel):
    """
    TẦNG 2: Động cơ thực thi toàn diện cho bài toán Object Detection.
    Đảm nhiệm toàn bộ phần cứng, vòng đời, suy luận batching, ProgressBar và xuất kết quả.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        unsupported: Optional[Union[Sequence[str], Set[str]]] = None,
        model: Optional[Any] = None,
        **kwargs,
    ) -> None:
        if metadata is None:
            model_name = "custom-detector"
            if model is not None:
                model_name = getattr(model, "name", getattr(model, "__name__", model.__class__.__name__))
            metadata = {
                "model_id": model_name,
                "backend": "PyTorch",
                "task": "Object-Detection",
            }
        super().__init__(metadata=metadata, unsupported=unsupported, **kwargs)
        self.task = "Object-Detection"
        self._device: str = "cpu"
        self._dtype: str = "float32"
        self.half_mode: bool = False
        self.model: Any = model
        self.processor: Any = None

    # =========================================================================
    # QUẢN LÝ PHẦN CỨNG & ĐỘ CHÍNH XÁC (Tầng 2 Detector Implementation)
    # =========================================================================
    @property
    def device(self) -> str:
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "hf_device_map"):
                return "multi-gpu"
            if hasattr(self.model, "parameters"):
                try:
                    return str(next(self.model.parameters()).device)
                except Exception:
                    pass
            if hasattr(self.model, "device"):
                return str(self.model.device)
        return self._device

    @property
    def dtype(self) -> str:
        if self.half_mode:
            return "float16"
        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "parameters"):
            try:
                p_dtype = str(next(self.model.parameters()).dtype)
                if "bfloat16" in p_dtype:
                    return "bfloat16"
                elif "float16" in p_dtype:
                    return "float16"
                elif "float32" in p_dtype:
                    return "float32"
            except Exception:
                pass
        return self._dtype

    def to(self, device_name: Union[str, int]) -> "Detector":
        self._check_supported("to")
        target_device = f"cuda:{device_name}" if isinstance(device_name, int) else str(device_name)
        self._device = target_device
        if hasattr(self, "model") and hasattr(self.model, "to"):
            # Nếu model đã được sharded qua device_map multi-gpu thì không gọi .to() đè
            if hasattr(self.model, "hf_device_map"):
                self.state = "MODIFIED"
                return self
            try:
                import torch
                self.model.to(target_device)
                if hasattr(self.model, "parameters"):
                    actual_dev = next(self.model.parameters()).device
                    self._device = str(actual_dev)
                    if actual_dev.type == "cpu" and next(self.model.parameters()).dtype == torch.float16:
                        self.model = self.model.float()
                        self.half_mode = False
                        self._dtype = "float32"
                from klygo import cuda
                if ("cuda" in self._device or self._device == "multi-gpu") and cuda.is_available() and self.half_mode:
                    if hasattr(self.model, "half"):
                        self.model = self.model.half()
                    self._dtype = "float16"
            except Exception:
                pass
        self.state = "MODIFIED"
        return self

    def cpu(self) -> "Detector":
        self._check_supported("cpu")
        self.to("cpu")
        self.float()
        return self

    def cuda(self, device: Optional[Union[int, str]] = None) -> "Detector":
        self._check_supported("cuda")
        target = "cuda" if device is None else (f"cuda:{device}" if isinstance(device, int) else str(device))
        return self.to(target)

    def half(self) -> "Detector":
        self._check_supported("half")
        self.half_mode = True
        self._dtype = "float16"
        if hasattr(self, "model"):
            try:
                from klygo import cuda
                if ("cuda" in str(self.device) or self.device == "multi-gpu") and cuda.is_available():
                    if hasattr(self.model, "half"):
                        self.model = self.model.half()
                    elif hasattr(self.model, "to"):
                        import torch
                        self.model = self.model.to(torch.float16)
                else:
                    if hasattr(self.model, "float"):
                        self.model = self.model.float()
            except Exception:
                pass
        self.state = "MODIFIED"
        return self

    def bfloat16(self) -> "Detector":
        self._check_supported("bfloat16")
        self.half_mode = False
        self._dtype = "bfloat16"
        if hasattr(self, "model"):
            try:
                import torch
                if hasattr(self.model, "bfloat16"):
                    self.model = self.model.bfloat16()
                elif hasattr(self.model, "to"):
                    self.model = self.model.to(torch.bfloat16)
            except Exception:
                pass
        self.state = "MODIFIED"
        return self

    def bfloat(self) -> "Detector":
        """Alias của bfloat16."""
        return self.bfloat16()

    def float(self) -> "Detector":
        self._check_supported("float")
        self.half_mode = False
        self._dtype = "float32"
        if hasattr(self, "model") and hasattr(self.model, "float"):
            self.model = self.model.float()
        self.state = "MODIFIED"
        return self

    # =========================================================================
    # VÒNG ĐỜI & BỘ NHỚ (Lifecycle & Resource Management)
    # =========================================================================
    def reset(self) -> "Detector":
        self._check_supported("reset")
        self.config = dict(self.default_config)
        self.cpu()
        self.state = "READY"
        return self

    def warmup(self) -> None:
        self._check_supported("warmup")
        dummy_img = PIL.Image.new("RGB", (1, 1), color="black")
        try:
            self.predict(source=dummy_img, prompt=["dummy"], verbose=False)
        except Exception:
            pass

    def clear_cache(self) -> None:
        self._check_supported("clear_cache")
        from klygo import cuda
        if cuda.is_available():
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def unload(self) -> None:
        self._check_supported("unload")
        self.cpu()
        self.clear_cache()
        if hasattr(self, "model"):
            del self.model
            self.model = None
        if hasattr(self, "processor"):
            del self.processor
            self.processor = None
        self.state = "UNLOADED"

    # =========================================================================
    # HỢP ĐỒNG AI LIFECYCLE CHUNG CHO DETECTION
    def train(self, mode: bool = True, *args, **kwargs) -> Any:
        """Chuyển chế độ huấn luyện (train mode) của PyTorch hoặc thực thi huấn luyện."""
        if args or (kwargs and not set(kwargs.keys()).issubset({"mode"})):
            self._check_supported("train")
            raise NotImplementedError(f"Mô hình '{self.model_id}' chưa hỗ trợ pipeline huấn luyện train() với bộ tham số này.")
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "train"):
                self.model.train(mode)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "train"):
                self.model.model.train(mode)
        return self

    def val(self, *args, **kwargs):
        self._check_supported("val")
        raise NotImplementedError(f"Mô hình '{self.model_id}' chưa hỗ trợ pipeline kiểm định val().")

    def export(self, output_dir: str) -> str:
        """Xuất toàn bộ mô hình (Weights + klygo.json) thành 1 thư mục Offline độc lập."""
        self._check_supported("export")
        from klygo import files
        abs_out = os.path.abspath(output_dir)
        files.mkdir(abs_out)

        # 1. Lưu processor và model weights
        if hasattr(self, "processor") and hasattr(self.processor, "save_pretrained"):
            try:
                self.processor.save_pretrained(abs_out)
            except Exception:
                pass

        if hasattr(self, "model"):
            if hasattr(self.model, "save_pretrained"):
                try:
                    self.model.save_pretrained(abs_out)
                except Exception:
                    pass
            elif hasattr(self.model, "save"):
                try:
                    self.model.save(os.path.join(abs_out, "model.pt"))
                except Exception:
                    pass

        # 2. Tạo và lưu file định danh klygo.json qua klygo.files
        meta_to_save = dict(self.metadata)
        meta_to_save["class"] = self.class_name
        meta_to_save["model_id"] = abs_out
        meta_to_save["config"] = self.config
        klygo_json_path = os.path.join(abs_out, "klygo.json")
        files.save(klygo_json_path, meta_to_save, verbose=False)

        return abs_out

    def save(self, output_dir: str) -> str:
        """Alias của export."""
        return self.export(output_dir)

    def benchmark(
        self,
        source: Optional[Any] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        iterations: int = 20,
        warmup: int = 5,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Đo đạc và đánh giá hiệu năng suy luận (Latency ms / FPS) của mô hình."""
        self._check_supported("benchmark")
        from klygo import cuda
        img = source if source is not None else PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        prompts = utils.normalize_prompt(prompt or ["object"])

        # 1. Warmup
        for _ in range(warmup):
            self.predict(source=img, prompt=prompts, verbose=False, **kwargs)

        # 2. Đo lường chính xác
        latencies = []
        is_gpu = cuda.is_available() and ("cuda" in str(self.device) or self.device == "multi-gpu")

        for _ in range(iterations):
            t_start = time.perf_counter()
            self.predict(source=img, prompt=prompts, verbose=False, **kwargs)
            if is_gpu:
                utils.cuda_sync()
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)

        avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
        fps = (1.0 / avg_lat) if avg_lat > 0 else 0.0
        w_dim, h_dim = (img.width, img.height) if isinstance(img, PIL.Image.Image) else (640, 640)

        report = {
            "model_id": self.model_id,
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "image_size": f"{w_dim}x{h_dim}",
            "iterations": iterations,
            "warmup": warmup,
            "latency_avg_ms": round(avg_lat * 1000, 2),
            "fps": round(fps, 1),
        }

        if verbose:
            print("=" * 60)
            print("         BAO CAO DANH GIA HIEU NANG & TOC DO MO HINH")
            print("=" * 60)
            print(f" * Mo hinh      : {report['model_id']}")
            print(f" * Backend/Task : {report['backend']} / {self.task}")
            print(f" * Thiet bi/Dtype: {report['device']} / {report['dtype']}")
            print(f" * So vong lap  : {report['iterations']} (Warmup: {report['warmup']})")
            print("-" * 60)
            print(f" * Do tre TB    : {report['latency_avg_ms']} ms / frame")
            print(f" * Toc do (FPS) : {report['fps']} FPS (frames / sec)")
            print("=" * 60)

        return report

    def help(self) -> None:
        """In ra hướng dẫn sử dụng chuẩn hóa cho mô hình Detection."""
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print(f"CLASS: {self.class_name}")
        print("=" * 60)
        print("1. predict(source, prompt, batch=1, vid_stride=1, max_frames=None, verbose=True, **kwargs)")
        print("   Nhan dien doi tuong tren anh, video, folder thong qua klygo.media.load.")
        print("2. benchmark(iterations=20, warmup=5)")
        print("   Danh gia toc do suy luan (Latency ms / FPS).")
        print("3. export(output_dir='my_offline_model')")
        print("   Xuat toan bo mo hinh thanh goi Offline doc lap.")

    # =========================================================================
    # ĐỘNG CƠ SUY LUẬN DETECTION HOÀN CHỈNH (predict & forward)
    # =========================================================================
    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: List[str],
        model_kwargs: Dict[str, Any],
        processor_kwargs: Dict[str, Any],
        post_kwargs: Dict[str, Any],
    ) -> List[Detection]:
        """Thực thi forward pass trên mô hình bên dưới."""
        if hasattr(self, "model") and callable(self.model):
            return self.model(images, prompt=prompt, **model_kwargs)
        raise NotImplementedError(f"Lớp '{self.class_name}' chưa triển khai hàm forward() cụ thể.")

    def __call__(self, *args, **kwargs) -> Any:
        """Cho phép gọi trực tiếp instance mô hình như một callable / PyTorch module."""
        if hasattr(self, "model") and callable(self.model):
            return self.model(*args, **kwargs)
        if hasattr(self, "forward"):
            return self.forward(*args, **kwargs)
        raise TypeError(f"'{type(self).__name__}' object is not callable.")
        """
        Phương thức suy luận AI cốt lõi: Nhận trực tiếp 3 gói kwargs:
        - processor_kwargs : Tiền xử lý
        - model_kwargs     : Suy luận AI
        - post_kwargs      : Hậu xử lý
        Lớp con cài đặt phương thức này.
        """
        raise NotImplementedError

    def predict(
        self,
        source: Any,
        prompt: Optional[Union[str, List[str]]] = None,
        batch: int = 1,
        vid_stride: int = 1,
        max_frames: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Detections:
        """
        Thực thi nhận diện đối tượng trên ảnh, video hoặc folder (luôn trả về tập hợp kết quả Detections).
        """
        self._check_supported("predict")

        # 1. Bắt buộc kiểm tra prompt
        actual_prompt = prompt or kwargs.pop("classes", None) or kwargs.pop("text_prompt", None)
        if actual_prompt is None:
            raise ValueError("Vui lòng cung cấp nhãn cần nhận diện qua tham số 'prompt'.")
        target_prompt = utils.normalize_prompt(actual_prompt)

        # 2. Phân giải nguồn dữ liệu qua klygo.media.load
        images, is_single = utils.resolve_images(source, step=vid_stride, max_frames=max_frames)
        if not images:
            return Detections(frames=[], source_type="list", fps=30.0)

        # 3. Phân giải 3 gói kwargs cho lần predict này
        model_kwargs, processor_kwargs, post_kwargs = utils.resolve_sub_kwargs(
            kwargs=kwargs,
            json_config=self.metadata.get("config"),
        )

        actual_batch = max(1, int(batch))

        # 4. Context inference mode tự động tăng tốc
        try:
            import torch
            infer_context = torch.inference_mode()
        except Exception:
            infer_context = utils.nullcontext()

        with infer_context:
            # A. Trường hợp 1 ảnh đơn lẻ
            if is_single:
                t_start = time.perf_counter()
                dets = self.forward(
                    images=images,
                    prompt=target_prompt,
                    model_kwargs=model_kwargs,
                    processor_kwargs=processor_kwargs,
                    post_kwargs=post_kwargs,
                )
                t_end = time.perf_counter()
                lat_ms = round((t_end - t_start) * 1000, 2)
                fps_val = round(1000.0 / max(0.001, lat_ms), 1)

                det = dets[0]
                if isinstance(det, dict):
                    cur_img = images[0]
                    box_objs = []
                    b_list = det.get("boxes", [])
                    s_list = det.get("scores", [1.0] * len(b_list))
                    l_list = det.get("labels", ["object"] * len(b_list))
                    for b_idx, (b, s, l) in enumerate(zip(b_list, s_list, l_list)):
                        box_objs.append(Box(id=b_idx, label=str(l), score=float(s), box=b, parent_image=cur_img))
                    det = Detection(
                        source_image=cur_img,
                        objects=box_objs,
                        image_frame_index=0,
                    )
                det.image_frame_index = 0
                det.speed = {"inference": lat_ms, "fps": fps_val}
                return Detections(frames=[det], source_type="image", fps=30.0)

            # B. Trường hợp Video / Folder / Batch ảnh
            frame_results = []
            with ProgressBar(total=len(images), desc="Predict", unit="frame", verbose=verbose, colour="cyan") as pbar:
                for i in range(0, len(images), actual_batch):
                    batch_imgs = images[i : i + actual_batch]
                    t_start = time.perf_counter()
                    dets = self.forward(
                        images=batch_imgs,
                        prompt=target_prompt,
                        model_kwargs=model_kwargs,
                        processor_kwargs=processor_kwargs,
                        post_kwargs=post_kwargs,
                    )
                    t_end = time.perf_counter()
                    lat_per_frame = round(((t_end - t_start) * 1000) / len(batch_imgs), 2)
                    fps_val = round(1000.0 / max(0.001, lat_per_frame), 1)

                    for idx, det in enumerate(dets):
                        if isinstance(det, dict):
                            cur_img = batch_imgs[idx] if idx < len(batch_imgs) else batch_imgs[0]
                            box_objs = []
                            b_list = det.get("boxes", [])
                            s_list = det.get("scores", [1.0] * len(b_list))
                            l_list = det.get("labels", ["object"] * len(b_list))
                            for b_idx, (b, s, l) in enumerate(zip(b_list, s_list, l_list)):
                                box_objs.append(Box(id=b_idx, label=str(l), score=float(s), box=b, parent_image=cur_img))
                            det = Detection(
                                source_image=cur_img,
                                objects=box_objs,
                                image_frame_index=i + idx,
                            )
                        det.image_frame_index = i + idx
                        det.speed = {"inference": lat_per_frame, "fps": fps_val}
                        frame_results.append(det)
                    pbar.update(len(batch_imgs))

            return Detections(frames=frame_results, source_type="video" if len(images) > 1 else "image", fps=30.0)
