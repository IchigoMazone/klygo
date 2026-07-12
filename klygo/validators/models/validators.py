from pathlib import Path
from typing import Any, Union, Tuple
from PIL import Image

from klygo.validators import validate_type

_SUPPORTED_MODELS = {"Grounding-Dino/232M", "Locate-Anything/3B"}


class InitModel:
    """Validate Kernel initialization parameters."""

    def __init__(self, model: str) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - model: Tham số model của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(model, str, "model")
        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported model: {model!r}, "
                f"supported models: {sorted(_SUPPORTED_MODELS)}"
            )
        self.model = model


class PredictModel:
    """Validate prediction parameters."""

    def __init__(
        self,
        images: Any,
        prompt: str,
        return_tensors: str,
        box_threshold: float,
        text_threshold: float,
        max_area: Union[float, None],
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - images: Tham số images của hàm
        - prompt: Tham số prompt của hàm
        - return_tensors: Tham số return_tensors của hàm
        - box_threshold: Tham số box_threshold của hàm
        - text_threshold: Tham số text_threshold của hàm
        - max_area: Tham số max_area của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(prompt, str, "prompt")
        validate_type(return_tensors, str, "return_tensors")
        validate_type(box_threshold, float, "box_threshold")
        validate_type(text_threshold, float, "text_threshold")

        if max_area is not None:
            validate_type(max_area, (int, float), "max_area")
            max_area = float(max_area)
            if not (0.0 <= max_area <= 1.0):
                raise ValueError(
                    "max_area must be between 0.0 and 1.0, "
                    f"got: {max_area}"
                )

        if not (0.0 <= box_threshold <= 1.0):
            raise ValueError(
                "box_threshold must be between 0.0 and 1.0, "
                f"got: {box_threshold}"
            )
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
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_area = max_area


class DetectModel:
    """Validate video detection parameters."""

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None],
        prompt: str,
        return_tensors: str,
        box_threshold: float,
        text_threshold: float,
        max_area: Union[float, None],
        desc: str,
        annotated_dir: Union[str, Path, None],
        dataset_dir: Union[str, Path, None],
        box_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        thickness: int,
        font: int,
        font_scale: float,
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - input_path: Đường dẫn file đầu vào
        - output_path: Đường dẫn file đầu ra
        - prompt: Tham số prompt của hàm
        - return_tensors: Tham số return_tensors của hàm
        - box_threshold: Tham số box_threshold của hàm
        - text_threshold: Tham số text_threshold của hàm
        - max_area: Tham số max_area của hàm
        - desc: Tham số desc của hàm
        - annotated_dir: Thư mục lưu các frame đã chú thích
        - dataset_dir: Thư mục lưu dataset
        - box_color: Tham số box_color của hàm
        - text_color: Tham số text_color của hàm
        - thickness: Tham số thickness của hàm
        - font: Tham số font của hàm
        - font_scale: Tham số font_scale của hàm

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(input_path, (str, Path), "input_path")
        validate_type(prompt, str, "prompt")
        validate_type(return_tensors, str, "return_tensors")
        validate_type(box_threshold, float, "box_threshold")
        validate_type(text_threshold, float, "text_threshold")
        if max_area is not None:
            validate_type(max_area, (int, float), "max_area")
            max_area = float(max_area)
            if not (0.0 <= max_area <= 1.0):
                raise ValueError(
                    "max_area must be between 0.0 and 1.0, "
                    f"got: {max_area}"
                )
        validate_type(desc, str, "desc")
        validate_type(thickness, int, "thickness")
        validate_type(font, int, "font")
        validate_type(font_scale, (int, float), "font_scale")

        for name, color in (("box_color", box_color), ("text_color", text_color)):
            validate_type(color, tuple, name)
            if len(color) != 3 or any(
                not isinstance(channel, int) or not 0 <= channel <= 255
                for channel in color
            ):
                raise ValueError(
                    f"{name} must contain three integers between 0 and 255, "
                    f"got: {color}"
                )

        if thickness <= 0:
            raise ValueError(f"thickness must be greater than 0, got: {thickness}")
        font_scale = float(font_scale)
        if font_scale <= 0:
            raise ValueError(f"font_scale must be greater than 0, got: {font_scale}")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"input_path must be a file, got: {input_path}")

        if output_path is not None:
            validate_type(output_path, (str, Path), "output_path")
            output_path = Path(output_path)
            if output_path.exists() and not output_path.is_file():
                raise ValueError(
                    f"output_path must be a file, got directory: {output_path}"
                )

        if annotated_dir is not None:
            validate_type(annotated_dir, (str, Path), "annotated_dir")
            annotated_dir = Path(annotated_dir)
            if annotated_dir.exists() and not annotated_dir.is_dir():
                raise ValueError(
                    f"annotated_dir must be a directory, got file: {annotated_dir}"
                )

        if dataset_dir is not None:
            validate_type(dataset_dir, (str, Path), "dataset_dir")
            dataset_dir = Path(dataset_dir)
            if dataset_dir.exists() and not dataset_dir.is_dir():
                raise ValueError(
                    f"dataset_dir must be a directory, got file: {dataset_dir}"
                )

        self.input_path = input_path
        self.output_path = output_path
        self.prompt = prompt
        self.return_tensors = return_tensors
        if not (0.0 <= box_threshold <= 1.0):
            raise ValueError(
                "box_threshold must be between 0.0 and 1.0, "
                f"got: {box_threshold}"
            )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_area = max_area
        self.desc = desc
        self.annotated_dir = annotated_dir
        self.dataset_dir = dataset_dir
        self.box_color = box_color
        self.text_color = text_color
        self.thickness = thickness
        self.font = font
        self.font_scale = font_scale
