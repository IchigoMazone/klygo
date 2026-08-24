import time
import warnings
from typing import Any, List, Dict, Union
import torch
import PIL.Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo.models import base, utils
from klygo.outputs.detect import Box, Detection

utils.suppress_ai_warnings()


class GroundingDinoDetect(base.Detector):
    """Trình bao bọc mô hình Zero-shot Object Detection kiến trúc Grounding DINO."""

    def __init__(self, metadata: Union[Dict[str, Any], str], **kwargs) -> None:
        super().__init__(metadata, **kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self._device).eval()

    def _infer_batch(
        self,
        batch_imgs: List[PIL.Image.Image],
        prompt: List[str],
        conf: float,
        text_threshold: float,
        half: bool,
    ) -> List[Detection]:
        """Xử lý suy luận cho 1 lô ảnh (hỗ trợ cả 1 ảnh đơn lẻ lẫn batch nhiều ảnh)."""
        t0 = time.perf_counter()
        inputs = self.processor(images=batch_imgs, text=[prompt] * len(batch_imgs), return_tensors="pt")
        dev = getattr(self.model, "device", self._device)
        dtype = next(self.model.parameters()).dtype

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device=dev, dtype=dtype if torch.is_floating_point(v) else None)

        t1 = time.perf_counter()
        use_half = "cuda" in str(dev) and (self.half_mode or half)
        with torch.no_grad():
            if use_half:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        if "cuda" in str(dev) and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        target_sizes = [img.size[::-1] for img in batch_imgs]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            batch_outs = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=conf,
                text_threshold=text_threshold,
            )

        t3 = time.perf_counter()
        speed = utils.calculate_speed(t0, t1, t2, t3, count=len(batch_imgs))

        results = []
        for out, img in zip(batch_outs, batch_imgs):
            labels = out.get("text_labels", out.get("labels", []))
            boxes = [
                Box(id=i, label=str(l), score=round(s.item(), 3), box=[round(x, 2) for x in b.tolist()], parent_image=img)
                for i, (b, s, l) in enumerate(zip(out["boxes"], out["scores"], labels))
            ]
            results.append(
                Detection(source_image=img, objects=boxes, speed=speed, text_prompt=prompt, threshold=conf, text_threshold=text_threshold)
            )
        return results

    def help(self) -> None:
        """In ra thông tin mô hình và danh sách các hàm nghiệp vụ."""
        print(f"MODEL: {self.model_id} ({self.backend}/{self.task})")
        print(f"CLASS: {self.class_name}")
        print("=" * 60)
        print("1. predict(source, prompt, conf=0.25, batch=1, vid_stride=1, max_frames=None, half=False, device=None)")
        print("   Nhan dien doi tuong tren anh, video, folder tu media.load.")
        print("2. benchmark(data='data.yaml', iterations=20, warmup=5)")
        print("   Cham diem danh gia toc do suy luan (Do tre Latency ms / Toc do FPS).")
        print("3. export(output_dir='my_model')")
        print("   Xuat toan bo mo hinh (Weights + klygo.json) ra thu muc de chay Offline.")
