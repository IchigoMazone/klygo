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
from klygo.models.backend import common
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
        """Device thực tế của model."""
        return common.current_device(self.model)

    def current_dtype(self) -> torch.dtype:
        """Dtype thực tế của model."""
        return common.current_dtype(self.model)

    # =========================================================================
    # BACKEND-AGNOSTIC EXECUTION HOOKS
    # =========================================================================
    def cast_inputs(self, inputs: Any) -> Any:
        """
        Tự động cast inputs (floating tensors) lên đúng device và dtype của model.
        Ủy thác hoàn toàn cho backend tương ứng.
        """
        return common.cast_inputs(self.backend, inputs, dev=self.current_device(), dtype=self.current_dtype())

    def run_inference(self, inputs: Any, **model_kwargs) -> Any:
        """
        Thực thi forward của model với AMP autocast và CUDA sync tự động.
        Ủy thác hoàn toàn cho backend tương ứng.
        """
        return common.run_inference(self.backend, self.model, inputs, cur_dtype=self.current_dtype(), **model_kwargs)

    def get_output_device(self, outputs: Any) -> torch.device:
        """
        Dò tìm device thực tế của kết quả đầu ra (logits, pred_boxes, tensor).
        Ủy thác hoàn toàn cho backend tương ứng.
        """
        return common.get_output_device(self.backend, outputs, default_device=self.current_device())

    def format_results(self, raw_outputs: Any) -> List[Dict[str, Any]]:
        """
        Chuẩn hóa kết quả thô của backend thành format [{boxes, scores, labels}].
        Ủy thác hoàn toàn cho backend tương ứng.
        """
        return common.format_results(self.backend, raw_outputs)

    # =========================================================================
    # CONVENIENCE HELPERS
    # =========================================================================
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
        common.reset(self.backend, self.model, getattr(self, "processor", None))
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
        common.clear_cache()

    def save(self, output_dir: str) -> None:
        """
        Lưu toàn bộ mô hình (Metadata + Python Code + Trọng số) ra thư mục chuẩn Klygo.
        Mô hình xuất ra có thể được nạp lại hoàn chỉnh qua `models.load(folder)`.
        Ủy thác hoàn toàn việc lưu trữ cho backend.common.save().
        """
        common.save(
            backend=self.backend,
            model=self.model,
            output_dir=output_dir,
            metadata=self.metadata,
            class_module=self.__class__.__module__,
            processor=getattr(self, "processor", None),
        )

    def unload(self) -> None:
        """
        Giải phóng tài nguyên và đưa trạng thái về UNLOADED.
        Ủy thác hoàn toàn việc thu hồi tài nguyên cho backend.common.unload().
        """
        common.unload(self.backend, self.model, getattr(self, "processor", None))
        if hasattr(self, "processor"):
            del self.processor
            self.processor = None
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
