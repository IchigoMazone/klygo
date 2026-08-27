"""
Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
TANG 3: Chi cai dat forward() dac thu — toan bo pipeline dung chung duoc Detector ho tro.
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
        self._load_hf_components(
            model_cls=AutoModelForZeroShotObjectDetection,
            processor_cls=AutoProcessor,
        )

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
        inputs = self._cast_inputs(self.processor(
            images=images, text=[labels] * len(images),
            return_tensors="pt", **processor_kwargs,
        ))

        # 2. Inference
        outputs = self._run_inference(inputs, **model_kwargs)

        # 3. Postprocess
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
        return self._build_detections(images, raw, prompt=prompt, threshold=thresh, text_threshold=text_thresh)
