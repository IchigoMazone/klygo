"""
Grounding DINO Zero-Shot Object Detection (klygo.models.detection.grounding_dino).
LAYER 3: Explicit model & processor configuration; ``forward()`` is concise thanks to Detector helpers.
"""

from typing import Any, List, Dict, Optional, Union
from collections.abc import Mapping
import PIL.Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError as exc:
    raise ImportError(
        "Grounding DINO requires the Hugging Face model extra. "
        "Install it using 'pip install \"klygo[transformers]\"'."
    ) from exc

from klygo import utils
from klygo.models.detection.base import Detector
from klygo.outputs import build_detections
from klygo.outputs.detect import Detection


class GroundingDinoDetect(Detector):
    """Zero-shot Object Detection — Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        super().__init__(
            metadata=metadata,
            **kwargs,
        )
        self._result_names = {}
        self._active_prompt = None

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

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]],
        native: Optional[bool] = None,
        **kwargs,
    ) -> List[Detection]:

        if not prompt:
            raise ValueError("Grounding DINO requires a non-empty text prompt; use reset() to clear saved state")

        model_config = dict(self.metadata["model"])
        processor_config = dict(self.metadata["processor"])
        post_config = dict(self.metadata["post"])

        proc_opts = {**processor_config, **kwargs}
        post_opts = {**post_config, **kwargs}

        # 2. In-place preprocessing (Grounding DINO-specific prompt formatting)
        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
        text_str = ". ".join(prompt_list) + "."
        text_batch = [text_str] * len(images)

        inputs = self.processor(
            images=images,
            text=text_batch,
            return_tensors="pt",
            **utils.exclude_keys(proc_opts, ["threshold", "text_threshold"]),
        )
        inputs = self.cast_inputs(inputs)

        # 3. Inference (AMP + automatic GPU sync via backend helpers)
        loading_keys = ["device_map", "torch_dtype", "dtype", "low_cpu_mem_usage", "cache_dir",
                        "revision", "local_files_only", "trust_remote_code", "token"]
        outputs = self.run_inference(inputs, **utils.exclude_keys(model_config, loading_keys))

        # 4. Grounding DINO postprocessing
        thresh = float(post_opts.get("threshold", 0.25))
        text_thresh = float(post_opts.get("text_threshold", 0.3))

        # Only sync 'input_ids' to the output device when needed (Multi-GPU).
        # Never copy large 'pixel_values' tensors; that would waste PCIe bandwidth and VRAM.
        target_dev = self.get_output_device(outputs)
        raw_input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        input_ids = self.sync_device(raw_input_ids, target_device=target_dev)

        with self.suppress_warnings():
            raw = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=input_ids,
                threshold=thresh,
                text_threshold=text_thresh,
                target_sizes=[img.size[::-1] for img in images],
                **utils.exclude_keys(post_opts, ["threshold", "text_threshold", "target_sizes"]),
            )

        # Filter out predictions with empty or whitespace-only labels from tokenizer
        cleaned_raw = []
        for item in raw:
            boxes = item.get("boxes", [])
            scores = item.get("scores", [])
            labels = item.get("labels", [])
            valid_indices = [
                i for i, lbl in enumerate(labels)
                if lbl is not None and str(lbl).strip()
            ]
            if len(valid_indices) == len(labels):
                cleaned_raw.append(item)
            else:
                cleaned_raw.append({
                    "boxes": [boxes[i] for i in valid_indices],
                    "scores": [scores[i] for i in valid_indices],
                    "labels": [str(labels[i]).strip() for i in valid_indices],
                })

        # 5. Pack into Detection objects
        results = build_detections(
            images,
            cleaned_raw,
            metadata={"prompt": prompt, "threshold": thresh, "text_threshold": text_thresh, "native": bool(native)},
            names=self._result_names,
        )
        if results:
            self._result_names = dict(results[0]._registry)
        return results
