"""
Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
LAYER 3: Explicit model & processor configuration; ``forward()`` is concise thanks to Detector helpers.
"""

from typing import Any, List, Dict, Union
import PIL.Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError as exc:
    raise ImportError(
        "Grounding DINO requires the Hugging Face model extra. "
        "Install it using 'pip install \"klygo[transformers]\"'."
    ) from exc

from klygo import outputs, utils
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection


class GroundingDinoDetect(Detector):
    """Zero-shot Object Detection — Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(
            metadata=metadata,
            **kwargs,
        )
        self._result_names = {}

        model_config = dict(self.metadata["model"])
        processor_config = dict(self.metadata["processor"])

        # 2. Initialise Processor & Model from Hugging Face
        with self.suppress_warnings():
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_config)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id,
                **model_config,
            )
            if "device_map" not in model_config:
                self.model.to("cpu")
            self.model.eval()

    def _prepare_prediction(self, prompt: List[str]) -> None:
        prompt = [
            str(name).strip().rstrip(".").lower()
            for name in (prompt or [])
            if str(name).strip().rstrip(".")
        ]
        self.names = {index: name for index, name in enumerate(prompt)}
        self._result_names = dict(self.names)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]],
    ) -> List[Detection]:
        model_config = dict(self.metadata["model"])
        processor_config = dict(self.metadata["processor"])
        post_config = dict(self.metadata["post"])

        # 2. In-place preprocessing (Grounding DINO-specific prompt formatting)
        raw_prompt = [prompt] if isinstance(prompt, str) else list(prompt) if prompt else []
        text_str = ". ".join([str(p).strip().rstrip(".").lower() for p in raw_prompt if str(p).strip()]) + "."
        text_batch = [text_str] * len(images) if text_str != "." else None

        inputs = self.processor(
            images=images,
            text=text_batch,
            return_tensors="pt",
            **processor_config,
        )
        inputs = self.cast_inputs(inputs)

        # 3. Inference (AMP + automatic GPU sync via backend helpers)
        outputs = self.run_inference(inputs, **model_config)

        # 4. Grounding DINO postprocessing
        thresh = post_config.get("threshold", 0.25)
        text_thresh = post_config.get("text_threshold", 0.3)

        # Only sync 'input_ids' to the output device when needed (Multi-GPU).
        # Never copy large 'pixel_values' tensors; that would waste PCIe bandwidth and VRAM.
        target_dev = self.get_output_device(outputs)
        raw_input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        input_ids = self.sync_device(raw_input_ids, target_device=target_dev)

        with self.suppress_warnings():
            raw = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=input_ids,
                threshold=thresh,
                text_threshold=text_thresh,
                target_sizes=[img.size[::-1] for img in images],
            **utils.exclude_keys(post_config, ["threshold", "text_threshold", "target_sizes"]),
            )

        # 5. Pack into Detection objects
        results = outputs.build_detections(
            images,
            raw,
            metadata={"prompt": prompt, **post_config},
            names=self._result_names,
        )
        if results:
            self._result_names = dict(results[0]._all_names)
        return results
