"""Optional NVIDIA LocateAnything-3B inference using its single-image API.

The upstream model generates box tokens without confidence scores. Klygo uses
score=1.0 as a documented presence value, never as measured confidence.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from klygo import files, outputs, validators
from .base import Detector

__all__ = ["LocateAnythingDetect"]

_MODEL_ID = "nvidia/LocateAnything-3B"
_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"

_BOX_PATTERN = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>")
_NONE_BOX_PATTERN = re.compile(r"<box>\s*None\s*</box>", re.IGNORECASE)

_GENERATION_DEFAULTS: Dict[str, Any] = {
    "generation_mode": "hybrid",
    "max_new_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": False,
}

_ALLOWED_LOADING_OPTIONS = {
    "trust_remote_code",
    "revision",
    "device",
    "torch_dtype",
    "cache_dir",
    "local_files_only",
}


def _extract_bounding_boxes(answer: Any, width: int, height: int) -> List[List[float]]:
    """Parse and scale coordinate tokens from Locate Anything model output.

    Parameters
    ----------
    answer : str
        Raw decoded generation text from the language model.
    width : int
        Source image width in pixels.
    height : int
        Source image height in pixels.

    Returns
    -------
    list of list of float
        Bounding boxes in absolute pixel coordinates [x1, y1, x2, y2].
    """
    validators.validate_type(answer, str, "Locate Anything response")

    box_text = _NONE_BOX_PATTERN.sub("None", answer)
    matches = list(_BOX_PATTERN.finditer(box_text))

    if box_text.count("<box>") != len(matches) or box_text.count("</box>") != len(matches):
        raise ValueError("Malformed or truncated Locate Anything box output")

    if not matches and not re.search(r"\bnone\b", answer, re.IGNORECASE):
        raise ValueError("Locate Anything returned neither boxes nor an explicit None result")

    boxes = []
    for match in matches:
        x1, y1, x2, y2 = map(int, match.groups())
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("Locate Anything returned invalid normalized coordinates")
        boxes.append([
            x1 * width / 1000.0,
            y1 * height / 1000.0,
            x2 * width / 1000.0,
            y2 * height / 1000.0,
        ])
    return boxes


class LocateAnythingDetect(Detector):
    """Load the pinned NVIDIA model with explicit remote-code consent.

    Parameters
    ----------
    metadata : mapping
        Klygo model metadata. The model group must include trust_remote_code=True.
        Only the NVIDIA checkpoint or a local copy is supported. This adapter
        requires CUDA and Transformers 4.57.1.
    **kwargs : object
        Detector construction options.

    Notes
    -----
    NVIDIA's non-commercial model license applies. Each image/category query
    runs sequentially. CPU, visual prompts, pointing and native batch decoding
    are unsupported. Importing this module does not import model dependencies.
    """

    def __init__(self, metadata: Any = None, **kwargs: Any) -> None:
        super().__init__(metadata=metadata, backend="Hugging Face", **kwargs)

        options = dict(self.metadata.get("model", {}))
        if options.get("trust_remote_code") is not True:
            raise ValueError(
                "Locate Anything requires explicit model_trust_remote_code=True "
                "to load NVIDIA's custom code"
            )

        unsupported_options = set(options) - _ALLOWED_LOADING_OPTIONS
        if unsupported_options:
            raise ValueError(
                f"Unsupported Locate Anything loading options: {sorted(unsupported_options)}"
            )

        if self.model_id != _MODEL_ID and not files.is_dir(self.model_id):
            raise ValueError(f"Use {_MODEL_ID!r} or a local copy of that checkpoint")

        revision = options.get("revision", _REVISION)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Locate Anything revision must be a full commit hash")

        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                'Install the integration with pip install "klygo[locate-anything]"'
            ) from error

        if transformers.__version__ != "4.57.1":
            raise ImportError(
                "Locate Anything requires transformers==4.57.1; "
                "use an isolated klygo[locate-anything] environment"
            )

        device = torch.device(options.get("device", "cuda"))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Locate Anything requires CUDA-enabled PyTorch and an NVIDIA GPU; "
                "CPU inference is unsupported by this adapter"
            )

        dtype = options.get("torch_dtype", torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""), None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("torch_dtype must be float16, bfloat16 or float32")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU lacks bfloat16 support; select model_torch_dtype='float16'")

        loading_kwargs = {
            "trust_remote_code": True,
            "revision": revision,
            "cache_dir": options.get("cache_dir", "artifacts/weights/locate-anything"),
            "local_files_only": options.get("local_files_only", False),
        }

        config = AutoConfig.from_pretrained(self.model_id, **loading_kwargs)
        if getattr(config, "model_type", None) != "locateanything":
            raise ValueError("Checkpoint config must have model_type='locateanything'")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **loading_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id, **loading_kwargs)

        for callback_name in ("py_apply_chat_template", "process_vision_info"):
            if not callable(getattr(self.processor, callback_name, None)):
                raise ValueError(f"Unsupported Locate Anything processor: missing {callback_name}")

        self.model = (
            AutoModel.from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=dtype,
                **loading_kwargs,
            )
            .to(device)
            .eval()
        )
        self._revision = revision

    def forward(
        self,
        images: Sequence[Image.Image],
        prompt: Any = None,
        native: Optional[bool] = None,
        **kwargs,
    ) -> List[Any]:
        """Generate real box tokens for each image/query and normalize coordinates."""
        # 1. Validate prompt and reject incompatible postprocessing overrides
        if not prompt:
            raise ValueError(
                "Locate Anything requires a non-empty prompt; use reset() to clear saved state"
            )
        if any("<" in label or ">" in label for label in prompt):
            raise ValueError("Locate Anything labels must not contain model control tokens")
        if self.metadata.get("post") or self.metadata.get("processor"):
            raise ValueError(
                "Locate Anything does not expose confidence thresholds or processor overrides"
            )

        # 2. Validate and build generation hyperparameters
        options = dict(self.metadata.get("generation", {}))
        merged_opts = {**options, **kwargs}
        unknown_options = set(merged_opts) - set(_GENERATION_DEFAULTS)
        if unknown_options:
            raise ValueError(
                f"Unsupported Locate Anything generation options: {sorted(unknown_options)}"
            )

        generation = {**_GENERATION_DEFAULTS, **merged_opts}
        if generation["generation_mode"] not in ("fast", "hybrid", "slow"):
            raise ValueError("generation_mode must be fast, hybrid or slow")

        validators.validate_count(generation["max_new_tokens"], "max_new_tokens")
        validators.validate_type(generation["do_sample"], bool, "do_sample")

        for key in ("temperature", "top_p", "repetition_penalty"):
            val = generation[key]
            validators.validate_type(val, (int, float), key)
            if isinstance(val, bool) or not math.isfinite(val) or val <= 0 or (key == "top_p" and val > 1):
                raise ValueError(f"Invalid generation option {key}")

        # 3. Sequential query execution per image and category
        predictions: List[Dict[str, List[Any]]] = []
        raw_answers: List[List[str]] = []

        for image in images:
            frame_pred: Dict[str, List[Any]] = {
                "boxes": [],
                "scores": [],
                "labels": [],
                "class_ids": [],
            }
            image_responses: List[str] = []

            for class_id, label in self.names.items():
                question = f"Locate all the instances that matches the following description: {label}."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                templated_text = self.processor.py_apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                vision_pictures, vision_videos = self.processor.process_vision_info(messages)

                encoded_inputs = self.processor(
                    text=[templated_text],
                    images=vision_pictures,
                    videos=vision_videos,
                    return_tensors="pt",
                )
                inputs = self.cast_inputs(encoded_inputs)

                response = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=self.tokenizer,
                    use_cache=True,
                    verbose=False,
                    **generation,
                )

                answer = response[0] if isinstance(response, tuple) else response
                image_responses.append(answer)

                extracted_boxes = _extract_bounding_boxes(answer, image.width, image.height)
                for box in extracted_boxes:
                    frame_pred["boxes"].append(box)
                    frame_pred["scores"].append(1.0)
                    frame_pred["labels"].append(label)
                    frame_pred["class_ids"].append(class_id)

            predictions.append(frame_pred)
            raw_answers.append(image_responses)

        # 4. Construct standardized Detection frames and decorate presence metadata
        frames = outputs.build_detections(
            images,
            predictions,
            names=self.names,
            metadata={
                "prompt": prompt,
                "score_kind": "presence_only",
                "confidence_available": False,
                "revision": self._revision,
                "generation": generation,
                "native": bool(native),
            },
        )

        for frame, answers in zip(frames, raw_answers):
            frame.metadata["raw_answers"] = answers
            for box in frame:
                box.metadata.update(score_kind="presence_only", native_score=None)

        return frames
