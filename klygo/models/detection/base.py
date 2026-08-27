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
import torch
import torch.nn as nn
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
    # QUAN LY PHAN CUNG & DO CHINH XAC (Detector Implementation)
    # nn.Module la Core: cuda/half/float/to/eval/train deu goi super() thang
    # Detector chi can: guard _check_supported + sync state + multi-gpu guard
    # =========================================================================
    def _sync_state(self) -> None:
        """Dong bo _device va _dtype tu trang thai thuc te cua nn.Module parameters."""
        try:
            params = list(nn.Module.parameters(self))
            if not params:
                return
            p = params[0]
            self._device = str(p.device)
            dtype_str = str(p.dtype)
            if "bfloat16" in dtype_str:
                self._dtype = "bfloat16"
                self.half_mode = False
            elif "float16" in dtype_str:
                self._dtype = "float16"
                self.half_mode = True
            else:
                self._dtype = "float32"
        except Exception:
            pass

    def _is_multi_gpu(self) -> bool:
        """Kiem tra xem inner model co dang dung HF device_map multi-GPU khong."""
        inner = self.__dict__.get("_modules", {}).get("model", None)
        return inner is not None and hasattr(inner, "hf_device_map")

    @property
    def device(self) -> str:
        if self._is_multi_gpu():
            return "multi-gpu"
        try:
            params = list(nn.Module.parameters(self))
            if params:
                return str(params[0].device)
        except Exception:
            pass
        inner = self.__dict__.get("_modules", {}).get("model", None)
        if inner is not None and hasattr(inner, "device"):
            return str(inner.device)
        return self._device

    @property
    def dtype(self) -> str:
        if self.half_mode:
            return "float16"
        try:
            params = list(nn.Module.parameters(self))
            if params:
                dtype_str = str(params[0].dtype)
                if "bfloat16" in dtype_str:
                    return "bfloat16"
                elif "float16" in dtype_str:
                    return "float16"
        except Exception:
            pass
        return self._dtype

    def to(self, *args, **kwargs) -> "Detector":
        """Chuyen model sang device/dtype chi dinh. Ho tro ca Klygo-style (int) va PyTorch-style."""
        self._check_supported("to")
        # Guard: HF multi-GPU sharding khong duoc goi .to()
        if self._is_multi_gpu():
            self.state = "MODIFIED"
            return self
        # Ho tro Klygo-style: to(0) -> to("cuda:0")
        if len(args) == 1 and isinstance(args[0], int):
            args = ("cuda:{}".format(args[0]),)
        # Goi nn.Module.to() that su — xu ly toan bo submodule tu dong
        nn.Module.to(self, *args, **kwargs)
        # Neu chuyen sang CPU voi FP16 -> tu dong ve FP32 de tranh loi
        try:
            params = list(nn.Module.parameters(self))
            if params and params[0].device.type == "cpu" and params[0].dtype == torch.float16:
                nn.Module.float(self)
                self.half_mode = False
        except Exception:
            pass
        self._sync_state()
        self.state = "MODIFIED"
        return self

    def cpu(self) -> "Detector":
        """Chuyen model ve CPU."""
        self._check_supported("cpu")
        nn.Module.cpu(self)      # Di chuyen toan bo submodule ve CPU
        nn.Module.float(self)   # CPU khong ho tro FP16 inference -> auto float32
        self.half_mode = False
        self._sync_state()
        self.state = "MODIFIED"
        return self

    def cuda(self, device=None) -> "Detector":
        """Chuyen model len GPU CUDA chi dinh."""
        self._check_supported("cuda")
        if self._is_multi_gpu():
            self.state = "MODIFIED"
            return self
        nn.Module.cuda(self, device)   # Di chuyen toan bo submodule len GPU
        # Khoi phuc half_mode neu dang bat nhung chua o float16
        if self.half_mode:
            nn.Module.half(self)
        self._sync_state()
        self.state = "MODIFIED"
        return self

    def half(self) -> "Detector":
        """Chuyen model sang FP16 (chi ap dung that su tren GPU)."""
        self._check_supported("half")
        self.half_mode = True
        self._dtype = "float16"
        from klygo import cuda as klygo_cuda
        if "cuda" in str(self.device) and klygo_cuda.is_available():
            nn.Module.half(self)   # Ap dung FP16 that su tren GPU
        else:
            # CPU: ghi nhan half_mode nhung giu FP32 de tranh loi
            nn.Module.float(self)
        self.state = "MODIFIED"
        return self

    def bfloat16(self) -> "Detector":
        """Chuyen model sang BF16."""
        self._check_supported("bfloat16")
        self.half_mode = False
        self._dtype = "bfloat16"
        nn.Module.to(self, torch.bfloat16)
        self.state = "MODIFIED"
        return self

    def bfloat(self) -> "Detector":
        """Alias cua bfloat16."""
        return self.bfloat16()

    def float(self) -> "Detector":
        """Chuyen model ve FP32."""
        self._check_supported("float")
        self.half_mode = False
        self._dtype = "float32"
        nn.Module.float(self)   # Chuyen toan bo submodule ve float32
        self.state = "MODIFIED"
        return self

    # =========================================================================
    # VONG DOI & BO NHO (Lifecycle & Resource Management)
    # =========================================================================
    def reset(self) -> "Detector":
        self._check_supported("reset")
        self._settings = dict(self._default_settings)  # Reset Klygo runtime settings
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
    # AI LIFECYCLE CHUNG CHO DETECTION
    # =========================================================================
    def val(self, *args, **kwargs):
        self._check_supported("val")
        raise NotImplementedError("Model '{}' chua ho tro pipeline val().".format(self.model_id))

    def export(self, output_dir: str) -> str:
        """Xuat toan bo mo hinh (Weights + klygo.json) thanh 1 thu muc Offline doc lap."""
        self._check_supported("export")
        from klygo import files
        abs_out = os.path.abspath(output_dir)
        files.mkdir(abs_out)

        # 1. Luu processor va model weights
        if hasattr(self, "processor") and hasattr(self.processor, "save_pretrained"):
            try:
                self.processor.save_pretrained(abs_out)
            except Exception:
                pass

        if hasattr(self, "model") and self.model is not None:
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

        # 2. Tao va luu file dinh danh klygo.json
        # Dung self.settings (Klygo dict) thay vi self.config (co the la HF PretrainedConfig)
        meta_to_save = dict(self.metadata)
        meta_to_save["class"] = self.class_name
        meta_to_save["model_id"] = abs_out
        meta_to_save["config"] = self.settings
        klygo_json_path = os.path.join(abs_out, "klygo.json")
        files.save(klygo_json_path, meta_to_save, verbose=False)

        return abs_out

    def save(self, output_dir: str) -> str:
        """Alias cua export."""
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
    def __call__(self, *args, **kwargs) -> Any:
        """Cho phép gọi trực tiếp instance mô hình:
        - Nếu truyền Tensor -> Gọi thẳng nn.Module bên dưới (Chuẩn PyTorch thuần).
        - Nếu truyền ảnh/đường dẫn/prompt -> Gọi predict() (Chuẩn Klygo Engine).
        """
        if args and not isinstance(args[0], (PIL.Image.Image, str, list, tuple)):
            import sys
            if "torch" in sys.modules:
                import torch
                if isinstance(args[0], torch.Tensor):
                    if hasattr(self, "model") and callable(self.model):
                        return self.model(*args, **kwargs)
        if "prompt" in kwargs or (args and isinstance(args[0], (str, PIL.Image.Image, list))):
            return self.predict(*args, **kwargs)
        if hasattr(self, "model") and callable(self.model):
            return self.model(*args, **kwargs)
        if hasattr(self, "forward"):
            return self.forward(*args, **kwargs)
        raise TypeError(f"'{type(self).__name__}' object is not callable.")

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
