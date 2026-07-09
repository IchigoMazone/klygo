from pathlib import Path
from typing import Any, Union, List, Tuple
from PIL import Image

from klygo.validators import validate_type

_SUPPORTED_MODELS = {"Grounding-Dino/232M", "Locate-Anything/3B"}


class InitKernel:
    """Validate Kernel initialization parameters."""

    def __init__(self, model: str) -> None:
        validate_type(model, str, "model")
        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported model: {model!r}, "
                f"supported models: {sorted(_SUPPORTED_MODELS)}"
            )
        self.model = model


class PredictKernel:
    """Validate prediction parameters."""

    def __init__(
        self,
        images: Any,
        prompt: str,
        return_tensors: str,
        threshold: float,
        text_threshold: float,
    ) -> None:
        validate_type(prompt, str, "prompt")
        validate_type(return_tensors, str, "return_tensors")
        validate_type(threshold, float, "threshold")
        validate_type(text_threshold, float, "text_threshold")

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0, got: {threshold}")
        if not (0.0 <= text_threshold <= 1.0):
            raise ValueError(f"text_threshold must be between 0.0 and 1.0, got: {text_threshold}")

        # Normalize images to list and validate they are PIL Images
        if not isinstance(images, (list, tuple)):
            img_list = [images]
        else:
            img_list = list(images)

        for i, img in enumerate(img_list):
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f"images must be PIL.Image.Image or sequence of PIL Images, "
                    f"item at index {i} is {type(img)}"
                )

        self.images = img_list
        self.prompt = prompt
        self.return_tensors = return_tensors
        self.threshold = threshold
        self.text_threshold = text_threshold


class DetectKernel:
    """Validate video detection parameters."""

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None],
        prompt: str,
        return_tensors: str,
        threshold: float,
        text_threshold: float,
        desc: str,
        save_frame_dir: Union[str, Path, None],
        save_yolo_dir: Union[str, Path, None],
    ) -> None:
        validate_type(input_path, (str, Path), "input_path")
        validate_type(prompt, str, "prompt")
        validate_type(return_tensors, str, "return_tensors")
        validate_type(threshold, float, "threshold")
        validate_type(text_threshold, float, "text_threshold")
        validate_type(desc, str, "desc")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"input_path must be a file, got: {input_path}")

        if output_path is not None:
            validate_type(output_path, (str, Path), "output_path")
            output_path = Path(output_path)

        if save_frame_dir is not None:
            validate_type(save_frame_dir, (str, Path), "save_frame_dir")
            save_frame_dir = Path(save_frame_dir)

        if save_yolo_dir is not None:
            validate_type(save_yolo_dir, (str, Path), "save_yolo_dir")
            save_yolo_dir = Path(save_yolo_dir)

        self.input_path = input_path
        self.output_path = output_path
        self.prompt = prompt
        self.return_tensors = return_tensors
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.desc = desc
        self.save_frame_dir = save_frame_dir
        self.save_yolo_dir = save_yolo_dir
