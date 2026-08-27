"""
Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
TANG 3: Cau hinh model & processor ro rang, forward() ngan gon nho cac helper cua Detector.
"""

from typing import Any, List, Dict, Union
import PIL.Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class GroundingDinoDetect(Detector):
    """Zero-shot Object Detection — Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(metadata=metadata, unsupported=("train", "val"), **kwargs)

        # 1. Boc tach 3 nhom cau hinh tu metadata
        mod_kw, proc_kw, _ = self.parse_config()

        # 2. Khoi tao Processor & Model tu Hugging Face
        with self.suppress_warnings():
            self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_kw)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **mod_kw)
            if "device_map" not in mod_kw:
                self.model.to(self._device)
            self.model.eval()

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> List[Detection]:
        # 1. Boc tach 3 nhom kwargs tu **kwargs
        mod_kw, proc_kw, post_kw = self.split_kwargs(kwargs)

        # 2. Preprocess (chuan hoa prompt + processor + device casting gon trong 1 dong)
        inputs = self.process_inputs(images, prompt, **proc_kw)

        # 3. Inference (AMP, GPU sync tu dong)
        outputs = self.run_inference(inputs, **mod_kw)

        # 4. Dong bo Multi-GPU va Postprocess dac thu cua Grounding DINO
        thresh = post_kw.get("threshold", 0.25)
        text_thresh = post_kw.get("text_threshold", 0.3)
        aligned_inputs = self.align_inputs_with_outputs(inputs, outputs)

        with self.suppress_warnings():
            raw = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=aligned_inputs.get("input_ids"),
                threshold=thresh,
                text_threshold=text_thresh,
                target_sizes=[img.size[::-1] for img in images],
                **self.filter_kwargs(post_kw, "threshold", "text_threshold", "target_sizes"),
            )

        # 5. Pack output
        return self.build_detections(images, raw, prompt=prompt, **post_kw)
