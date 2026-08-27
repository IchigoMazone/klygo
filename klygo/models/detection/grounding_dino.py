"""
Trình bao bọc mô hình Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
TẦNG 3: Concrete Model kế thừa Detector, khóa các tính năng không hỗ trợ và cài đặt forward().
"""

import warnings
from typing import Any, List, Dict
import torch
import PIL.Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo.models import utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Box, Detection

utils.suppress_ai_warnings()


class GroundingDinoDetect(Detector):
    """Mô hình nhận diện Zero-shot Object Detection kiến trúc Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(
            metadata=metadata,
            unsupported=("train", "val"),
            **kwargs,
        )

        cfg = self.metadata.get("config", {})
        proc_kw = dict(cfg.get("processor", {}))
        mod_kw = dict(cfg.get("model", {}))

        # Tự động chuyển đổi torch_dtype string -> torch.dtype
        if "torch_dtype" in mod_kw:
            dt = mod_kw["torch_dtype"]
            if isinstance(dt, str):
                dt_lower = dt.lower()
                if dt_lower in ("bfloat16", "bf16"):
                    import torch
                    mod_kw["torch_dtype"] = torch.bfloat16
                    self._dtype = "bfloat16"
                elif dt_lower in ("float16", "fp16", "half"):
                    import torch
                    mod_kw["torch_dtype"] = torch.float16
                    self._dtype = "float16"
                    self.half_mode = True
                elif dt_lower in ("float32", "fp32"):
                    import torch
                    mod_kw["torch_dtype"] = torch.float32
                    self._dtype = "float32"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_kw)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **mod_kw)
            if "device_map" not in mod_kw:
                self.model.to(self._device)
            self.model.eval()

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: List[str],
        model_kwargs: Dict[str, Any],
        processor_kwargs: Dict[str, Any],
        post_kwargs: Dict[str, Any],
    ) -> List[Detection]:
        """
        Động cơ suy luận tinh khiết:
        - 1. Tiền xử lý với processor_kwargs
        - 2. Forward với model_kwargs
        - 3. Hậu xử lý với post_kwargs
        """
        # Luôn luôn lấy device thực tế từ parameter đầu tiên của mô hình
        dev = None
        if hasattr(self.model, "parameters"):
            try:
                dev = next(self.model.parameters()).device
            except Exception:
                pass
        if dev is None:
            dev = getattr(self.model, "device", None) or self._device

        # 1. Tiền xử lý (Định dạng List[List[str]] chuẩn Transformers mới)
        labels_list = [p.strip().rstrip(".").lower() for p in prompt if p.strip()]
        text_labels = [labels_list] * len(images)
        inputs = self.processor(
            images=images,
            text=text_labels,
            return_tensors="pt",
            **processor_kwargs,
        )

        import torch
        target_dtype = None
        if hasattr(self.model, "parameters"):
            try:
                target_dtype = next(self.model.parameters()).dtype
            except Exception:
                pass

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and target_dtype in (torch.bfloat16, torch.float16):
                    inputs[k] = v.to(device=dev, dtype=target_dtype)
                else:
                    inputs[k] = v.to(device=dev)

        # 2. Suy luận AI
        from klygo import cuda
        is_gpu = ("cuda" in str(dev) or self.device == "multi-gpu") and cuda.is_available()
        use_half = is_gpu and self.half_mode
        dev_type = "cuda" if is_gpu else "cpu"
        effective_dtype = "float16" if use_half else ("bfloat16" if target_dtype == torch.bfloat16 else "float32")

        with utils.amp_autocast_if_needed(use_half=use_half, dtype=effective_dtype, device_type=dev_type):
            outputs = self.model(**inputs, **model_kwargs)

        if is_gpu:
            utils.cuda_sync()

        # 3. Hậu xử lý (Sử dụng trực tiếp threshold & text_threshold chuẩn)
        thresh = post_kwargs.get("threshold", 0.25)
        text_thresh = post_kwargs.get("text_threshold", 0.3)
        target_sizes = [img.size[::-1] for img in images]

        post_proc_kwargs = {
            "outputs": outputs,
            "input_ids": inputs.input_ids,
            "threshold": thresh,
            "text_threshold": text_thresh,
            "target_sizes": target_sizes,
        }
        for k, v in post_kwargs.items():
            if k not in post_proc_kwargs:
                post_proc_kwargs[k] = v

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw_results = self.processor.post_process_grounded_object_detection(
                **post_proc_kwargs
            )

        # 4. Đóng gói kết quả
        results = []
        for res, img in zip(raw_results, images):
            boxes = [
                Box(
                    id=i,
                    label=str(label),
                    score=round(float(score.item()), 3),
                    box=[round(float(x), 2) for x in box.tolist()],
                    parent_image=img,
                )
                for i, (score, label, box) in enumerate(zip(res["scores"], res["labels"], res["boxes"]))
            ]
            results.append(
                Detection(
                    source_image=img,
                    objects=boxes,
                    prompt=prompt,
                    threshold=thresh,
                    text_threshold=text_thresh,
                )
            )
        return results
