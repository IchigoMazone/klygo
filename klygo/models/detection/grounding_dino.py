"""Grounding DINO with explicit configuration, preprocessing, and inference."""

from typing import Any, List, Dict, Optional, Union
from collections.abc import Mapping
import PIL.Image

try:
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError as exc:
    raise ImportError(
        "Grounding DINO requires the Hugging Face model extra. "
        "Install it using 'pip install \"klygo[transformers]\"'."
    ) from exc

from klygo import utils
from klygo.models.detection.base import Detector
from klygo.models.utils import normalize_prompt
from klygo.outputs import build_detections
from klygo.outputs.detect import Detection


class GroundingDinoDetect(Detector):
    """Zero-shot Object Detection — Grounding DINO."""

    def __init__(self, metadata: Dict[str, Any], **kwargs) -> None:
        kwargs.setdefault("backend", "Hugging Face")
        super().__init__(
            metadata=metadata,
            **kwargs,
        )
        self._active_prompt = None

        model_config = dict(self.metadata["model"])
        processor_config = dict(self.metadata["processor"])
        weights = model_config.pop("weights", None)

        # Initialise Processor & Model from Hugging Face
        with self.suppress_warnings():
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_config)
            model_architecture = None
            if weights:
                loading_keys = (
                    "cache_dir",
                    "revision",
                    "local_files_only",
                    "trust_remote_code",
                    "token",
                )
                config_options = {
                    key: model_config[key]
                    for key in loading_keys
                    if key in model_config
                }
                model_architecture = AutoConfig.from_pretrained(
                    self.model_id,
                    **config_options,
                )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                weights or self.model_id,
                config=model_architecture,
                **model_config,
            )
            if "device_map" not in model_config:
                self.model.to("cpu")
            self.model.eval()

    def cast_inputs(self, inputs):
        """Move floating inputs while preserving integer token tensors."""
        from klygo.models.detection import _huggingface as execution

        return execution.cast_inputs(inputs, dev=self.current_device(), dtype=self.current_dtype())

    def run_inference(self, inputs, **model_kwargs):
        """Execute this model using its native Transformers input structure."""
        from klygo.models.detection import _huggingface as execution

        return execution.run_inference(self.model, inputs, cur_dtype=self.current_dtype(), **model_kwargs)

    def get_output_device(self, outputs):
        """Find the actual output shard for multi-device inference."""
        from klygo.models.detection import _huggingface as execution

        return execution.get_output_device(outputs, default_device=self.current_device())

    def sync_device(self, tensor, target_device):
        """Move postprocessing tokens to the output shard when needed."""
        from klygo.models.detection import _huggingface as execution

        return execution.sync_device(tensor, target_device)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> List[Detection]:

        if not prompt:
            raise ValueError("Grounding DINO requires a non-empty text prompt")

        prompt_list = normalize_prompt(prompt)
        self.names = {class_id: label for class_id, label in enumerate(prompt_list)}
        self._active_prompt = list(prompt_list)

        model_config = dict(self.metadata["model"])
        processor_config = dict(self.metadata["processor"])
        post_config = dict(self.metadata["post"])

        proc_opts = {**processor_config, **kwargs}
        post_opts = {**post_config, **kwargs}

        # Format prompts and tokenize text batch
        text_str = ". ".join(prompt_list) + "."
        text_batch = [text_str] * len(images)

        inputs = self.processor(
            images=images,
            text=text_batch,
            return_tensors="pt",
            **utils.exclude_keys(proc_opts, ["threshold", "text_threshold"]),
        )
        inputs = self.cast_inputs(inputs)

        # Loading-only options never belong to the model's forward call.
        loading_keys = [
            "device_map", "torch_dtype", "dtype", "low_cpu_mem_usage", "cache_dir",
            "revision", "local_files_only", "trust_remote_code", "token", "weights",
        ]
        model_options = utils.exclude_keys(model_config, loading_keys)
        outputs = self.model(**inputs, **model_options)

        thresh = float(post_opts.get("threshold", 0.25))
        text_thresh = float(post_opts.get("text_threshold", 0.3))

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

        # Filter out empty or whitespace-only labels
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

        # Pack into Detection objects
        results = build_detections(
            images,
            cleaned_raw,
            metadata={
                "prompt": prompt_list,
                "threshold": thresh,
                "text_threshold": text_thresh,
            },
            names=self.names,
        )
        if results:
            self.names = dict(results[0]._registry)

        return results
