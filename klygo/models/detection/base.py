"""
Lớp cơ sở chuyên biệt cho bài toán Nhận diện Đối tượng (klygo.models.detection.base).
TẦNG 2: Task Engine Interface + Implementation - Cài đặt toàn bộ Động cơ Detection,
phân giải đa nguồn qua media.load, Batching loop, ProgressBar, đo thời gian ngoài luồng,
quản lý phần cứng, export và benchmark.
"""

import os
import time
from abc import abstractmethod
from typing import Dict, Any, Optional, Union, Sequence, Set, List, Tuple
import torch
import torch.nn as nn
import PIL.Image

from klygo.models.base import BaseModel
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
        flags: Optional[Sequence[str]] = None,
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
        
        if flags is None:
            raise ValueError(f"Bắt buộc phải khai báo 'flags' (ví dụ: flags=('model', 'post')) khi khởi tạo mô hình {self.__class__.__name__} để đảm bảo sự tường minh.")

        super().__init__(metadata=metadata, flags=flags, unsupported=unsupported, **kwargs)
        self.task = "Object-Detection"
        self._device: str = "cpu"
        self._dtype: str = "float32"
        self.half_mode: bool = False
        self.model: Any = model
        self.processor: Any = None

    @property
    def device(self) -> str:
        try:
            if hasattr(self.model, "parameters"):
                params = list(self.model.parameters())
                if params:
                    return str(params[0].device)
        except Exception:
            pass
        return str(getattr(self.model, "device", self._device))

    @property
    def dtype(self) -> str:
        try:
            if hasattr(self.model, "parameters"):
                params = list(self.model.parameters())
                if params:
                    dtype_str = str(params[0].dtype)
                    if "bfloat16" in dtype_str:
                        return "bfloat16"
                    elif "float16" in dtype_str:
                        return "float16"
        except Exception:
            pass
        return self._dtype


    # =========================================================================
    # PUBLIC HELPERS CHO MODEL IMPLEMENTATION (Tầng 3)
    # =========================================================================
    DTYPE_MAP = {
        "float16": (torch.float16, "float16", True),
        "fp16": (torch.float16, "float16", True),
        "half": (torch.float16, "float16", True),
        "bfloat16": (torch.bfloat16, "bfloat16", False),
        "bf16": (torch.bfloat16, "bfloat16", False),
    }
    _DTYPE_MAP = DTYPE_MAP

    def parse_dtype(self, dt_str: str):
        """Map chuỗi định dạng dtype sang tuple: (torch.dtype, dtype_str, half_mode)."""
        return self.DTYPE_MAP.get(str(dt_str).lower(), (torch.float32, "float32", False))

    def resolve_dtype(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Tự động chuẩn hóa torch_dtype trong kwargs và đồng bộ state của model."""
        dt = kwargs.get("torch_dtype")
        if isinstance(dt, str):
            kwargs["torch_dtype"], self._dtype, self.half_mode = self.parse_dtype(dt)
        return kwargs

    def parse_config(self, *groups: str) -> Tuple[Dict[str, Any], ...]:
        """
        Bóc tách các nhóm cấu hình từ self.metadata['config'].
        Mặc định sử dụng self._flags ('model', 'processor', 'post' cho Hugging Face)
        hoặc bất kỳ danh sách nhóm nào được truyền vào.
        """
        target_groups = groups if groups else getattr(self, "_flags", ("model", "processor", "post"))
        cfg = self.metadata.get("config", {})
        result = []
        for g in target_groups:
            val = dict(cfg.get(g, {}))
            if g == "model":
                val = self.resolve_dtype(val)
            result.append(val)
        return tuple(result)

    def split_kwargs(
        self,
        kwargs: Optional[Dict[str, Any]] = None,
        *groups: str,
        **extra_kwargs,
    ) -> Tuple[Dict[str, Any], ...]:
        """
        Bóc tách và hợp nhất các nhóm kwargs theo self._flags (hoặc groups truyền vào).
        Ví dụ: mod_kw, proc_kw, post_kw = self.split_kwargs(kwargs)
        """
        kw = dict(kwargs or {})
        kw.update(extra_kwargs)
        target_groups = groups if groups else getattr(self, "_flags", ("model", "processor", "post"))
        return utils.resolve_sub_kwargs(kwargs=kw, json_config=self.metadata.get("config"), groups=target_groups)

    def filter_kwargs(self, kwargs: Dict[str, Any], *exclude_keys: Union[str, Sequence[str]]) -> Dict[str, Any]:
        """
        Lọc bỏ các key đã xử lý khỏi dictionary kwargs để truyền an toàn các tham số còn lại.
        Ví dụ: self.filter_kwargs(post_kw, "threshold", "text_threshold", "target_sizes")
        """
        exclude_set = set()
        for item in exclude_keys:
            if isinstance(item, (tuple, list, set)):
                exclude_set.update(item)
            else:
                exclude_set.add(item)
        return {k: v for k, v in kwargs.items() if k not in exclude_set}



    def current_device(self) -> torch.device:
        """Device thực tế của model (lấy từ parameter đầu tiên)."""
        import torch
        try:
            if hasattr(self.model, "parameters"):
                return next(self.model.parameters()).device
        except StopIteration:
            pass
        return torch.device(self._device)

    def current_dtype(self) -> torch.dtype:
        """Dtype thực tế của model (lấy từ parameter đầu tiên)."""
        import torch
        try:
            if hasattr(self.model, "parameters"):
                return next(self.model.parameters()).dtype
        except StopIteration:
            pass
        return torch.float32

    def get_output_device(self, outputs: Any) -> torch.device:
        """Dò tìm thiết bị thực tế của tensor đầu ra (hỗ trợ ModelOutput, Dict, List, Tensor)."""
        if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
            return outputs.logits.device
        if hasattr(outputs, "pred_boxes") and isinstance(outputs.pred_boxes, torch.Tensor):
            return outputs.pred_boxes.device
        if isinstance(outputs, torch.Tensor):
            return outputs.device
        if isinstance(outputs, dict):
            for v in outputs.values():
                if isinstance(v, torch.Tensor):
                    return v.device
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item.device
                if isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, torch.Tensor):
                            return v.device
        return self.current_device()

    def cast_inputs(self, inputs):
        """
        Cast tất cả floating tensors trong inputs lên đúng device + dtype an toàn của model.
        - Trên CPU: Luôn giữ Float32 để tránh lỗi mat1 and mat2 (CPU không hỗ trợ FP16).
        - Trên GPU: Tự động khớp FP16 / BF16 / FP32 với non_blocking=True.
        - Tensor số nguyên (input_ids, attention_mask): Giữ nguyên int64/bool.
        """
        dev = self.current_device()
        dtype = self.current_dtype()
        is_cpu = (dev.type == "cpu")
        is_cuda = (dev.type == "cuda")

        if hasattr(inputs, "keys"):
            for k in list(inputs.keys()):
                v = inputs[k]
                if not isinstance(v, torch.Tensor):
                    continue
                if v.is_floating_point():
                    if is_cpu:
                        inputs[k] = v.to(device=dev, dtype=torch.float32)
                    elif dtype in (torch.float16, torch.bfloat16):
                        inputs[k] = v.to(device=dev, dtype=dtype, non_blocking=is_cuda)
                    else:
                        inputs[k] = v.to(device=dev, dtype=torch.float32, non_blocking=is_cuda)
                else:
                    inputs[k] = v.to(device=dev, non_blocking=is_cuda)
        elif isinstance(inputs, (list, tuple)):
            casted = []
            for v in inputs:
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        if is_cpu:
                            casted.append(v.to(device=dev, dtype=torch.float32))
                        elif dtype in (torch.float16, torch.bfloat16):
                            casted.append(v.to(device=dev, dtype=dtype, non_blocking=is_cuda))
                        else:
                            casted.append(v.to(device=dev, dtype=torch.float32, non_blocking=is_cuda))
                    else:
                        casted.append(v.to(device=dev, non_blocking=is_cuda))
                else:
                    casted.append(v)
            inputs = type(inputs)(casted)
        elif isinstance(inputs, torch.Tensor):
            if inputs.is_floating_point():
                if is_cpu:
                    inputs = inputs.to(device=dev, dtype=torch.float32)
                elif dtype in (torch.float16, torch.bfloat16):
                    inputs = inputs.to(device=dev, dtype=dtype, non_blocking=is_cuda)
                else:
                    inputs = inputs.to(device=dev, dtype=torch.float32, non_blocking=is_cuda)
            else:
                inputs = inputs.to(device=dev, non_blocking=is_cuda)

        return inputs

    def run_inference(self, inputs, **model_kwargs):
        """
        Thực thi forward của self.model với AMP autocast và device sync tự động.
        """
        cur_dt = self.current_dtype()
        use_half = (cur_dt == torch.float16) or self.half_mode

        if cur_dt == torch.bfloat16:
            eff_dtype = "bfloat16"
        elif use_half:
            eff_dtype = "float16"
        else:
            eff_dtype = "float32"

        with utils.amp_autocast_if_needed(use_half=use_half, dtype=eff_dtype):
            if hasattr(inputs, "items") or isinstance(inputs, dict):
                outputs = self.model(**inputs, **model_kwargs)
            elif isinstance(inputs, (list, tuple)):
                outputs = self.model(*inputs, **model_kwargs)
            else:
                outputs = self.model(inputs, **model_kwargs)

        utils.cuda_sync()
        return outputs

    def build_detections(
        self,
        images: List[PIL.Image.Image],
        raw_results: List[Dict[str, Any]],
        **metadata,
    ) -> List[Detection]:
        """
        Chuẩn hóa danh sách kết quả raw [{'scores', 'labels', 'boxes'}] thành List[Detection].
        Nhận linh hoạt mọi metadata (prompt, threshold, text_threshold, iou, nms, ...) qua **metadata.
        Ví dụ: self.build_detections(images, raw, prompt=prompt, **post_kw)
        """
        results = []
        for res, img in zip(raw_results, images):
            boxes = []
            scores = res.get("scores", [])
            labels = res.get("labels", [])
            raw_boxes = res.get("boxes", [])
            for i, (score, label, box) in enumerate(zip(scores, labels, raw_boxes)):
                b_list = box.tolist() if hasattr(box, "tolist") else list(box)
                s_val = float(score.item()) if hasattr(score, "item") else float(score)
                boxes.append(
                    Box(
                        id=i,
                        label=str(label),
                        score=round(s_val, 3),
                        box=[round(float(x), 2) for x in b_list],
                        parent_image=img,
                    )
                )
            results.append(
                Detection(
                    source_image=img,
                    objects=boxes,
                    **metadata,
                )
            )
        return results



    # =========================================================================
    # VONG DOI & BO NHO (Lifecycle & Resource Management)
    # =========================================================================
    def reset(self) -> "Detector":
        self._settings = dict(self._default_settings)
        self.state = "READY"
        self.cpu()
        self.state = "READY"
        return self

    def warmup(self) -> None:
        """Chạy thử 1 lần với ảnh 640x640 để khởi động GPU pipeline trước khi dùng thực."""
        dummy_img = PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        try:
            self.predict(source=dummy_img, prompt=["object"], verbose=False)
        except Exception:
            pass

    def clear_cache(self) -> None:
        from klygo import cuda
        if cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def save(self, output_dir: str) -> None:
        """
        Lưu toàn bộ mô hình (Metadata + Python Code + Trọng số) ra thư mục chuẩn Klygo.
        Mô hình xuất ra có thể được nạp lại hoàn chỉnh qua `models.load(folder)`.
        """
        from klygo import files
        import shutil
        import sys
        
        abs_out = os.path.abspath(output_dir)
        files.mkdir(abs_out)
        
        # 1. Ghi klygo.json
        meta = dict(self.metadata)
        meta.pop("num_params", None)
        files.save(os.path.join(abs_out, "klygo.json"), meta, verbose=False)
        
        # 2. Xử lý Custom Class (copy model.py)
        module_name = self.__class__.__module__
        if not module_name.startswith("klygo.models."):
            mod = sys.modules.get(module_name)
            if mod and hasattr(mod, "__file__") and mod.__file__:
                source_file = mod.__file__
                if os.path.exists(source_file):
                    shutil.copy2(source_file, os.path.join(abs_out, "model.py"))
        
        # 3. Trọng số & Artifacts (chỉ base fallback)
        inner = self._inner_model()
        if inner is not None:
            if hasattr(inner, "save_pretrained"):
                inner.save_pretrained(abs_out)
        if hasattr(self, "processor") and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(abs_out)

    def unload(self) -> None:
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
        from klygo import cuda
        img = source if source is not None else PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        prompts = utils.normalize_prompt(prompt or ["object"])

        for _ in range(warmup):
            self.predict(source=img, prompt=prompts, verbose=False, **kwargs)

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
        """In ra huong dan su dung chuan hoa va cac tham so dac thu cua mo hinh."""
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print(f"CLASS: {self.class_name}")
        print("=" * 60)
        print("1. predict(source, prompt, batch=1, vid_stride=1, max_frames=None, verbose=True, **kwargs)")
        print("   Nhan dien doi tuong tren anh, video, folder thong qua klygo.media.load.")

        settings_dict = self.settings
        has_params = any(bool(v) for v in settings_dict.values()) if isinstance(settings_dict, dict) else False
        if has_params:
            print("\n   [Tham so dac thu co the tuy chinh trong predict]:")
            for group_name, group_val in settings_dict.items():
                if isinstance(group_val, dict) and group_val:
                    for k, v in group_val.items():
                        print(f"     * {k}={v} (nhom: {group_name})")

        print("\n2. benchmark(iterations=20, warmup=5)")
        print("   Danh gia toc do suy luan (Latency ms / FPS).")
        print("=" * 60)

    # =========================================================================
    # ĐỘNG CƠ SUY LUẬN DETECTION HOÀN CHỈNH (predict & forward)
    # =========================================================================
    def _pack_detection(
        self,
        det: Any,
        image: PIL.Image.Image,
        frame_index: int,
        lat_ms: float,
        fps_val: float,
    ) -> "Detection":
        """Chuẩn hóa 1 kết quả thô (dict hoặc Detection) thành Detection đầy đủ."""
        if isinstance(det, dict):
            b_list = det.get("boxes", [])
            s_list = det.get("scores", [1.0] * len(b_list))
            l_list = det.get("labels", ["object"] * len(b_list))
            box_objs = [
                Box(id=i, label=str(l), score=float(s), box=b, parent_image=image)
                for i, (b, s, l) in enumerate(zip(b_list, s_list, l_list))
            ]
            det = Detection(source_image=image, objects=box_objs, image_frame_index=frame_index)
        det.image_frame_index = frame_index
        det.speed = {"inference": lat_ms, "fps": fps_val}
        return det

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> List[Detection]:
        """Thực thi forward pass trên mô hình bên dưới."""
        mod_kw, _, _ = self.split_kwargs(kwargs)
        if hasattr(self, "model") and callable(self.model):
            return self.model(images, prompt=prompt, **mod_kw)
        return []

    def __call__(self, *args, **kwargs) -> Any:
        """Cho phép gọi trực tiếp instance mô hình:
        - Nếu truyền Tensor -> Gọi thẳng nn.Module bên dưới (Chuẩn PyTorch thuần).
        - Nếu truyền ảnh/đường dẫn/prompt -> Gọi predict() (Chuẩn Klygo Engine).
        """
        with utils.suppress_warnings():
            if args and not isinstance(args[0], (PIL.Image.Image, str, list, tuple)):
                if "torch" in __import__("sys").modules:
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
        stream: bool = False,
        **kwargs,
    ) -> Union[Detections, Any]:
        """
        Thực thi nhận diện đối tượng trên ảnh, video hoặc folder.
        Nếu stream=True, trả về Generator[Detection] (Chống tràn RAM video lớn).
        Nếu stream=False, trả về Detections (RAM tiêu chuẩn).
        """
        target_prompt = utils.normalize_prompt(prompt) if prompt is not None else None
        actual_batch = max(1, int(batch))

        try:
            infer_context = torch.inference_mode()
        except Exception:
            infer_context = utils.nullcontext()

        # ==========================================
        # LUỒNG STREAMING (CHỐNG TRÀN RAM)
        # ==========================================
        if stream:
            images, _ = utils.resolve_images(source, step=vid_stride, max_frames=max_frames, stream=True)
            
            def _stream_generator():
                import itertools
                iterator = iter(images)
                idx_offset = 0
                with infer_context, utils.suppress_warnings():
                    while True:
                        batch_imgs = list(itertools.islice(iterator, actual_batch))
                        if not batch_imgs:
                            break
                        
                        t_start = time.perf_counter()
                        dets = self.forward(images=batch_imgs, prompt=target_prompt, **kwargs)
                        t_end = time.perf_counter()
                        
                        lat_per_frame = round(((t_end - t_start) * 1000) / len(batch_imgs), 2)
                        fps_val = round(1000.0 / max(0.001, lat_per_frame), 1)
                        
                        for idx, det in enumerate(dets):
                            img = batch_imgs[idx] if idx < len(batch_imgs) else batch_imgs[0]
                            yield self._pack_detection(det, img, idx_offset + idx, lat_per_frame, fps_val)
                            
                        idx_offset += len(batch_imgs)
            return _stream_generator()

        # ==========================================
        # LUỒNG TIÊU CHUẨN (LƯU VÀO RAM Detections)
        # ==========================================
        images, is_single = utils.resolve_images(source, step=vid_stride, max_frames=max_frames, stream=False)
        if not images:
            return Detections(frames=[], source_type="list", fps=30.0)

        with infer_context, utils.suppress_warnings():
            if is_single:
                t_start = time.perf_counter()
                dets = self.forward(images=images, prompt=target_prompt, **kwargs)
                t_end = time.perf_counter()
                lat_ms = round((t_end - t_start) * 1000, 2)
                fps_val = round(1000.0 / max(0.001, lat_ms), 1)
                det = self._pack_detection(dets[0], images[0], 0, lat_ms, fps_val)
                return Detections(frames=[det], source_type="image", fps=30.0)

            frame_results = []
            with ProgressBar(total=len(images), desc="Predict", unit="frame", verbose=verbose, colour="cyan") as pbar:
                for i in range(0, len(images), actual_batch):
                    batch_imgs = images[i : i + actual_batch]
                    t_start = time.perf_counter()
                    dets = self.forward(images=batch_imgs, prompt=target_prompt, **kwargs)
                    t_end = time.perf_counter()
                    lat_per_frame = round(((t_end - t_start) * 1000) / len(batch_imgs), 2)
                    fps_val = round(1000.0 / max(0.001, lat_per_frame), 1)
                    for idx, det in enumerate(dets):
                        img = batch_imgs[idx] if idx < len(batch_imgs) else batch_imgs[0]
                        frame_results.append(self._pack_detection(det, img, i + idx, lat_per_frame, fps_val))
                    pbar.update(len(batch_imgs))

            return Detections(frames=frame_results, source_type="video" if not is_single else "image", fps=30.0)
