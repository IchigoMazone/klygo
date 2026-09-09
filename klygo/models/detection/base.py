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
import PIL.Image

from klygo.models.base import BaseModel
from klygo.models import utils
from klygo.outputs.detect import Detections, Detection, Box
from klygo.utils.progress import ProgressBar


class Detector(BaseModel):
    """
    TẦNG 2: Động cơ thực thi toàn diện cho bài toán Object Detection.
    Đảm nhiệm vòng đời suy luận batching, ProgressBar và đóng gói Detections.
    Hoàn toàn framework-agnostic, đọc toàn bộ thông tin cấu hình từ metadata.
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        flags: Sequence[str],
        unsupported: Optional[Union[Sequence[str], Set[str]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(metadata=metadata, flags=flags, unsupported=unsupported, **kwargs)
        self.model: Any = None


    def parse_config(self, *groups: str) -> Tuple[Dict[str, Any], ...]:
        """
        Bóc tách các nhóm cấu hình từ self.settings (mặc định lấy từ metadata['config']).
        Sử dụng self.flags hoặc bất kỳ danh sách nhóm nào được truyền vào.
        """
        target_groups = groups if groups else self.flags
        cfg = self.settings
        return tuple(dict(cfg.get(g, {})) for g in target_groups)

    def split_kwargs(
        self,
        kwargs: Optional[Dict[str, Any]] = None,
        *groups: str,
        **extra_kwargs,
    ) -> Tuple[Dict[str, Any], ...]:
        """
        Bóc tách và hợp nhất 2 tầng tham số theo self.flags (hoặc groups truyền vào):
        - Tầng 1: Cấu hình mặc định (từ metadata['config'] / self.settings)
        - Tầng 2: Runtime kwargs truyền vào khi gọi predict() / forward()
        """
        kw = dict(kwargs or {})
        kw.update(extra_kwargs)
        target_groups = groups if groups else self.flags
        return utils.resolve_sub_kwargs(
            kwargs=kw,
            json_config=self.settings,
            groups=target_groups,
            warn_unmatched=True,
        )

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
        except (StopIteration, Exception):
            pass
        if hasattr(self.model, "device"):
            return torch.device(self.model.device)
        return torch.device("cpu")

    def current_dtype(self) -> torch.dtype:
        """Dtype thực tế của model (lấy từ parameter đầu tiên)."""
        import torch
        try:
            if hasattr(self.model, "parameters"):
                return next(self.model.parameters()).dtype
        except (StopIteration, Exception):
            pass
        return torch.float32

    # =========================================================================
    # HUGGING FACE BACKEND HELPERS (hf_*)
    # =========================================================================
    def hf_cast_inputs(self, inputs):
        """
        Cast các floating tensors trong inputs (Hugging Face BatchFeature / Dict)
        lên đúng device + dtype an toàn của model (giữ nguyên int64 cho input_ids).
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

    def hf_run_inference(self, inputs, **model_kwargs):
        """
        Thực thi forward của Hugging Face model với AMP autocast và device sync tự động.
        """
        cur_dt = self.current_dtype()
        use_half = (cur_dt == torch.float16)

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

    def hf_get_output_device(self, outputs: Any) -> torch.device:
        """Dò tìm thiết bị thực tế của tensor đầu ra Hugging Face ModelOutput."""
        if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
            return outputs.logits.device
        if hasattr(outputs, "pred_boxes") and isinstance(outputs.pred_boxes, torch.Tensor):
            return outputs.pred_boxes.device
        if isinstance(outputs, torch.Tensor):
            return outputs.device
        return self.current_device()

    def hf_save(self, output_dir: str) -> None:
        """Lưu model & processor theo chuẩn Hugging Face save_pretrained()."""
        abs_out = os.path.abspath(output_dir)
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(abs_out)
        if hasattr(self, "processor") and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(abs_out)

    def hf_unload(self) -> None:
        """Dọn dẹp processor và giải phóng tài nguyên của Hugging Face."""
        if hasattr(self, "processor"):
            del self.processor
            self.processor = None

    # =========================================================================
    # ULTRALYTICS BACKEND HELPERS (ul_*)
    # =========================================================================
    def ul_format_results(self, ultra_results: Any) -> List[Dict[str, Any]]:
        """Bóc tách boxes, scores, labels từ kết quả trả về của Ultralytics YOLO."""
        raw_list = []
        if ultra_results is not None:
            for res in ultra_results:
                boxes_data, scores_data, labels_data = [], [], []
                if getattr(res, "boxes", None) is not None:
                    for b in res.boxes:
                        boxes_data.append(b.xyxy[0].tolist())
                        scores_data.append(float(b.conf[0].item()))
                        cls_id = int(b.cls[0].item())
                        labels_data.append(res.names.get(cls_id, str(cls_id)))
                raw_list.append({"boxes": boxes_data, "scores": scores_data, "labels": labels_data})
        return raw_list

    def ul_save(self, output_dir: str) -> None:
        """Lưu trọng số theo chuẩn Ultralytics YOLO."""
        abs_out = os.path.abspath(output_dir)
        if self.model is not None and hasattr(self.model, "save"):
            self.model.save(abs_out)

    def ul_unload(self) -> None:
        """Dọn dẹp tài nguyên Ultralytics."""
        pass

    # =========================================================================
    # CONVENIENCE ALIASES
    # =========================================================================
    def cast_inputs(self, inputs):
        return self.hf_cast_inputs(inputs)

    def run_inference(self, inputs, **model_kwargs):
        return self.hf_run_inference(inputs, **model_kwargs)

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
        if hasattr(self.model, "cpu"):
            self.model.cpu()
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
        Tự động bọc hf_save() hoặc ul_save() theo backend.
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
        
        # 3. Trọng số & Artifacts (bọc theo backend)
        if self.backend == "Hugging Face" or hasattr(self, "processor"):
            self.hf_save(abs_out)
        elif self.backend == "Ultralytics":
            self.ul_save(abs_out)
        elif self.model is not None:
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(abs_out)
            elif hasattr(self.model, "save"):
                self.model.save(abs_out)

    def unload(self) -> None:
        """
        Giải phóng tài nguyên và đưa trạng thái về UNLOADED.
        Tự động bọc hf_unload() hoặc ul_unload() theo backend.
        """
        if hasattr(self.model, "cpu"):
            self.model.cpu()
        self.clear_cache()
        if self.backend == "Hugging Face" or hasattr(self, "processor"):
            self.hf_unload()
        elif self.backend == "Ultralytics":
            self.ul_unload()
        if hasattr(self, "model"):
            del self.model
            self.model = None
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
        cur_dev = self.current_device()
        cur_dt = self.current_dtype()
        is_gpu = cuda.is_available() and (cur_dev.type == "cuda")

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
            "device": str(cur_dev),
            "dtype": str(cur_dt).replace("torch.", ""),
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
