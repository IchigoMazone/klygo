"""
Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
TANG 3: Cau hinh model & processor ro rang, forward() ngan gon nho cac helper cua Detector.
"""

import warnings
from typing import Any, List, Dict
import PIL.Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo.models import utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection

utils.suppress_ai_warnings()


class GroundingDinoDetect(Detector):
    """Zero-shot Object Detection — Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, unsupported=("train", "val"), **kwargs)

        # 1. Boc tach 3 nhom cau hinh ro rang (mod_kw tu dong chuan hoa torch_dtype & sync state)
        mod_kw, proc_kw, _ = self.parse_config()

        # 2. Khoi tao Processor & Model tu Hugging Face
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
        # 1. Preprocess & device casting
        labels = [p.strip().rstrip(".").lower() for p in prompt if p.strip()]
        inputs = self.cast_inputs(self.processor(
            images=images, text=[labels] * len(images),
            return_tensors="pt", **processor_kwargs,
        ))

        # 2. Inference (AMP, GPU sync tu dong)
        outputs = self.run_inference(inputs, **model_kwargs)

        # 3. Postprocess dac thu cua Grounding DINO
        thresh = post_kwargs.get("threshold", 0.25)
        text_thresh = post_kwargs.get("text_threshold", 0.3)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                threshold=thresh,
                text_threshold=text_thresh,
                target_sizes=[img.size[::-1] for img in images],
                **{k: v for k, v in post_kwargs.items() if k not in ("threshold", "text_threshold", "target_sizes")}
            )

        # 4. Pack output
        return self.build_detections(images, raw, prompt=prompt, threshold=thresh, text_threshold=text_thresh)
