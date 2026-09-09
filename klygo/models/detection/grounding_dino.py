"""
Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
TANG 3: Cau hinh model & processor ro rang, forward() ngan gon nho cac helper cua Detector.
"""

from typing import Any, List, Dict, Union, Sequence
import torch
import PIL.Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class GroundingDinoDetect(Detector):
    """Zero-shot Object Detection — Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(
            metadata=metadata,
            flags=("model", "processor", "post"),
            unsupported=("train", "val"),
            **kwargs,
        )

        # 1. Boc tach 3 nhom cau hinh tu metadata
        mod_kw, proc_kw, _ = self.parse_config()

        # 2. Khoi tao Processor & Model tu Hugging Face
        with self.suppress_warnings():
            self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_kw)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **mod_kw)
            if "device_map" not in mod_kw:
                self.model.to("cpu")
            self.model.eval()

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> List[Detection]:
        # 1. Boc tach 3 nhom kwargs tu **kwargs
        mod_kw, proc_kw, post_kw = self.split_kwargs(kwargs)

        # 2. Tiền xử lý tại chỗ (Đặc thù riêng của Grounding DINO)
        raw_prompt = [prompt] if isinstance(prompt, str) else list(prompt) if prompt else []
        text_str = ". ".join([str(p).strip().rstrip(".").lower() for p in raw_prompt if str(p).strip()]) + "."
        text_batch = [text_str] * len(images) if text_str != "." else None

        inputs = self.processor(
            images=images,
            text=text_batch,
            return_tensors="pt",
            **proc_kw
        )
        inputs = self.cast_inputs(inputs)

        # 3. Inference (AMP, GPU sync tu dong)
        outputs = self.run_inference(inputs, **mod_kw)

        # 4. Postprocess đặc thù của Grounding DINO
        thresh = post_kw.get("threshold", 0.25)
        text_thresh = post_kw.get("text_threshold", 0.3)

        # Tường minh: Chỉ đồng bộ duy nhất 'input_ids' về cùng device của outputs khi cần thiết (Multi-GPU).
        # Tuyệt đối không sao chép thừa thãi các tensor ảnh lớn (pixel_values) làm nghẽn PCIe / hao phí VRAM.
        target_dev = self.get_output_device(outputs)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        if isinstance(input_ids, torch.Tensor) and input_ids.device != target_dev:
            input_ids = input_ids.to(target_dev, non_blocking=(target_dev.type == "cuda"))

        with self.suppress_warnings():
            raw = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=input_ids,
                threshold=thresh,
                text_threshold=text_thresh,
                target_sizes=[img.size[::-1] for img in images],
                **self.filter_kwargs(post_kw, "threshold", "text_threshold", "target_sizes"),
            )

        # 5. Pack output
        return self.build_detections(images, raw, prompt=prompt, **post_kw)
