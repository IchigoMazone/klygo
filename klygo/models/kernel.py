import os
import re
from pathlib import Path
from typing import Any, Union, List, Dict, Tuple

import cv2 as cv
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

from klygo.validators.models import InitModel, PredictModel, DetectModel
from klygo.io import write_yaml


class Model:
    """A deep learning helper class for object detection and prediction.

    Supports zero-shot object detection with models like Grounding DINO. Can be
    used for processing individual images or detecting objects in video frames.
    """

    NAME_MODELS = {
        "GDN": "Grounding-Dino/232M",
        "LAG": "Locate-Anything/3B",
    }
    MODEL_HUGGING_FACE = {
        "GDN": "IDEA-Research/grounding-dino-base",
        "LAG": "",
    }

    def __init__(self, model: str = "Grounding-Dino/232M") -> None:
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
        - NotImplementedError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        params = InitModel(model=model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name_models = self.NAME_MODELS
        self.model_hugging_face = self.MODEL_HUGGING_FACE

        self.model = self._model(params.model)
        self.processor = self._processor(params.model)

        self.prompt = None
        self.return_tensors = None
        self.box_threshold = None
        self.text_threshold = None
        self.max_area = None
        self.input_kwargs: Dict[str, Any] = {}
        self.output_kwargs: Dict[str, Any] = {}

    def _processor(self, model: str) -> Any:
        """
        Tác dụng:
        - Thực hiện chức năng _processor

        Đầu vào:
        - self: Đối tượng hiện tại
        - model: Tham số model của hàm

        Đầu ra:
        - Kết quả xử lý của hàm

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - NotImplementedError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        if model == self.name_models.get("GDN"):
            return AutoProcessor.from_pretrained(
                self.model_hugging_face.get("GDN")
            )
        elif model == self.name_models.get("LAG"):
            raise NotImplementedError(
                f"Model {model!r} is not yet implemented."
            )
        else:
            raise ValueError(f"Unknown model name: {model}")

    def _model(self, model: str) -> Any:
        """
        Tác dụng:
        - Thực hiện chức năng _model

        Đầu vào:
        - self: Đối tượng hiện tại
        - model: Tham số model của hàm

        Đầu ra:
        - Kết quả xử lý của hàm

        Ngoại lệ:
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - NotImplementedError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        if model == self.name_models.get("GDN"):
            return AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_hugging_face.get("GDN")
            ).to(self.device)
        elif model == self.name_models.get("LAG"):
            raise NotImplementedError(
                f"Model {model!r} is not yet implemented."
            )
        else:
            raise ValueError(f"Unknown model name: {model}")

    def _imwrite_txt(self, path: Union[str, Path], data: str) -> None:
        """
        Tác dụng:
        - Thực hiện chức năng _imwrite_txt

        Đầu vào:
        - self: Đối tượng hiện tại
        - path: Đường dẫn file
        - data: Dữ liệu cần xử lý

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - OSError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(data)

    def set_input_kwargs(self, **kwargs: Any) -> None:
        """
        Tác dụng:
        - Thực hiện chức năng set_input_kwargs

        Đầu vào:
        - self: Đối tượng hiện tại
        - **kwargs: Các tham số bổ sung

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        self.input_kwargs = kwargs

    def set_output_kwargs(self, **kwargs: Any) -> None:
        """
        Tác dụng:
        - Thực hiện chức năng set_output_kwargs

        Đầu vào:
        - self: Đối tượng hiện tại
        - **kwargs: Các tham số bổ sung

        Đầu ra:
        - Không trả về dữ liệu

        Nguồn: TrinhNhuNhat_12072026.
        """
        self.output_kwargs = kwargs

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str = "apple.",
        return_tensors: str = "pt",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_area: Union[float, None] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tác dụng:
        - Nhận diện đối tượng trên một hoặc nhiều ảnh

        Đầu vào:
        - self: Đối tượng hiện tại
        - images: Tham số images của hàm
        - prompt: Tham số prompt của hàm
        - return_tensors: Tham số return_tensors của hàm
        - box_threshold: Tham số box_threshold của hàm
        - text_threshold: Tham số text_threshold của hàm
        - max_area: Tham số max_area của hàm

        Đầu ra:
        - Kết quả xử lý của hàm

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        params = PredictModel(
            images=images,
            prompt=prompt,
            return_tensors=return_tensors,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_area=max_area,
        )

        self.prompt = params.prompt
        self.return_tensors = params.return_tensors
        self.box_threshold = params.box_threshold
        self.text_threshold = params.text_threshold
        self.max_area = params.max_area
        target_size = [image.size[::-1] for image in params.images]

        inputs = self.processor(
            images=params.images,
            text=self.prompt,
            return_tensors=self.return_tensors,
            **self.input_kwargs,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_size,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            **self.output_kwargs,
        )

        if self.max_area is not None:
            for image, result in zip(params.images, results):
                boxes = result["boxes"]
                if len(boxes) == 0:
                    continue

                image_area = image.width * image.height
                box_areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
                    boxes[:, 3] - boxes[:, 1]
                ).clamp(min=0)
                keep = box_areas <= self.max_area * image_area
                for key in ("boxes", "scores", "labels"):
                    if key in result:
                        values = result[key]
                        if torch.is_tensor(values):
                            result[key] = values[keep]
                        else:
                            keep_list = keep.detach().cpu().tolist()
                            result[key] = [
                                value
                                for value, should_keep in zip(values, keep_list)
                                if should_keep
                            ]

        return results

    def crop(
        self,
        input_path: str,
        target: Union[str, None] = None,
        prompt: str = "apple.",
        return_tensors: str = "pt",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        desc: str = "Cropping",
        max_area: Union[float, None] = None,
        padding: int = 0,
    ) -> List[Image.Image]:
        """
        Tác dụng:
        - Nhận diện và cắt tất cả đối tượng từ từng frame của video

        Đầu vào:
        - self: Đối tượng hiện tại
        - input_path: Đường dẫn file video đầu vào
        - target: Thư mục lưu ảnh crop; None để trả trực tiếp ảnh PIL
        - prompt: Chuỗi mô tả các đối tượng cần nhận diện
        - return_tensors: Định dạng tensor đầu vào của model
        - box_threshold: Ngưỡng độ tin cậy của bounding box
        - text_threshold: Ngưỡng tương đồng giữa đối tượng và prompt
        - desc: Nội dung hiển thị trên thanh tiến trình
        - max_area: Tỉ lệ diện tích bounding box tối đa so với ảnh
        - padding: Số pixel mở rộng xung quanh bounding box

        Đầu ra:
        - Danh sách ảnh PIL của tất cả đối tượng đã cắt

        Ngoại lệ:
        - TypeError: Kiểu dữ liệu của tham số không hợp lệ
        - ValueError: padding là số âm hoặc tham số dự đoán không hợp lệ
        - FileNotFoundError: Video đầu vào không tồn tại hoặc không thể mở

        Nguồn: TrinhNhuNhat_12072026.
        """
        if not isinstance(padding, int):
            raise TypeError(
                f"padding must be int, got {type(padding).__name__}"
            )
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}")

        params = DetectModel(
            input_path=input_path,
            output_path=None,
            prompt=prompt,
            return_tensors=return_tensors,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_area=max_area,
            desc=desc,
            annotated_dir=None,
            dataset_dir=None,
            box_color=(255, 0, 0),
            text_color=(255, 0, 0),
            thickness=2,
            font=cv.FONT_HERSHEY_SIMPLEX,
            font_scale=0.6,
        )
        target_dir = Path(target) if target is not None else None
        if target_dir is not None:
            if target_dir.exists() and not target_dir.is_dir():
                raise ValueError(
                    f"target must be a directory, got file: {target_dir}"
                )
            target_dir.mkdir(parents=True, exist_ok=True)

        cap = cv.VideoCapture(str(params.input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {params.input_path}")

        frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        outputs: List[Image.Image] = []
        frame_index = 0
        for _ in tqdm(
            range(frames),
            desc=params.desc,
            colour="blue",
            bar_format="{l_bar}{bar:30}{r_bar}",
        ):
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            result = self.predict(
                images=image,
                prompt=params.prompt,
                return_tensors=params.return_tensors,
                box_threshold=params.box_threshold,
                text_threshold=params.text_threshold,
                max_area=params.max_area,
            )[0]
            height, width = frame.shape[:2]

            for crop_index, (box, label) in enumerate(
                zip(result.get("boxes", []), result.get("labels", []))
            ):
                values = box.tolist() if hasattr(box, "tolist") else list(box)
                x1, y1, x2, y2 = values
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(width, int(x2) + padding)
                y2 = min(height, int(y2) + padding)
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped = frame[y1:y2, x1:x2]
                crop_image = Image.fromarray(
                    cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
                )
                outputs.append(crop_image)
                if target_dir is not None:
                    label_name = re.sub(
                        r"[^a-zA-Z0-9_-]+",
                        "_",
                        str(label),
                    ).strip("_")
                    label_name = label_name or "unknown"
                    label_dir = target_dir / label_name
                    label_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = (
                        label_dir
                        / f"frame_{frame_index}_crop_{crop_index}.jpg"
                    )
                    cv.imwrite(str(crop_path), cropped)

            frame_index += 1

        cap.release()
        return outputs

    def detect(
        self,
        input_path: str,
        output_path: Union[str, None] = None,
        prompt: str = "apple.",
        return_tensors: str = "pt",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        desc: str = "Processing",
        annotated_dir: Union[str, None] = None,
        dataset_dir: Union[str, None] = None,
        max_area: Union[float, None] = None,
        box_color: Tuple[int, int, int] = (255, 0, 0),
        text_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        font: int = cv.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
    ) -> List[Image.Image]:
        """
        Tác dụng:
        - Nhận diện đối tượng trên video và lưu kết quả

        Đầu vào:
        - self: Đối tượng hiện tại
        - input_path: Đường dẫn file đầu vào
        - output_path: Đường dẫn file đầu ra
        - prompt: Tham số prompt của hàm
        - return_tensors: Tham số return_tensors của hàm
        - box_threshold: Tham số box_threshold của hàm
        - text_threshold: Tham số text_threshold của hàm
        - desc: Tham số desc của hàm
        - annotated_dir: Thư mục lưu các frame đã chú thích
        - dataset_dir: Thư mục lưu dataset
        - max_area: Tham số max_area của hàm
        - box_color: Tham số box_color của hàm
        - text_color: Tham số text_color của hàm
        - thickness: Tham số thickness của hàm
        - font: Tham số font của hàm
        - font_scale: Tham số font_scale của hàm

        Đầu ra:
        - Danh sách frame PIL đã được vẽ bounding box và label

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        params = DetectModel(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            return_tensors=return_tensors,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_area=max_area,
            desc=desc,
            annotated_dir=annotated_dir,
            dataset_dir=dataset_dir,
            box_color=box_color,
            text_color=text_color,
            thickness=thickness,
            font=font,
            font_scale=font_scale,
        )

        cap = cv.VideoCapture(str(params.input_path))
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video file: {params.input_path}"
            )

        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        count = 0
        class_to_id: Dict[str, int] = {}
        annotated_images: List[Image.Image] = []
        out = None

        if params.output_path:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(
                str(params.output_path),
                fourcc,
                fps,
                (width, height),
            )

        if params.annotated_dir:
            params.annotated_dir.mkdir(parents=True, exist_ok=True)

        if params.dataset_dir:
            params.dataset_dir.mkdir(parents=True, exist_ok=True)
            (params.dataset_dir / "images").mkdir(parents=True, exist_ok=True)
            (params.dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

        # Unified colored progress bar matching klygo archive/io style
        for _ in tqdm(
            range(frames),
            desc=params.desc,
            colour="blue",
            bar_format="{l_bar}{bar:30}{r_bar}",
        ):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            results = self.predict(
                images=image,
                prompt=params.prompt,
                return_tensors=params.return_tensors,
                box_threshold=params.box_threshold,
                text_threshold=params.text_threshold,
                max_area=params.max_area,
            )

            yolo_lines = []
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"]
            ):
                x1, y1, x2, y2 = box.tolist()

                if box is not None:
                    cv.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        params.box_color,
                        params.thickness,
                    )
                    cv.putText(
                        frame,
                        f"{label} {score:.2f}",
                        (int(x1), int(y1) - 5),
                        params.font,
                        params.font_scale,
                        params.text_color,
                        params.thickness,
                    )

                if params.dataset_dir:
                    label_clean = label.strip().lower()

                    if label_clean not in class_to_id:
                        class_to_id[label_clean] = len(class_to_id)

                    class_id = class_to_id[label_clean]
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    box_w = (x2 - x1) / width
                    box_h = (y2 - y1) / height

                    yolo_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{box_w:.6f} {box_h:.6f}"
                    )

            annotated_images.append(
                Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            )

            if params.annotated_dir:
                cv.imwrite(
                    str(params.annotated_dir / f"frame_{count}.jpg"), frame
                )

            if params.dataset_dir:
                img_path = (
                    params.dataset_dir / "images" / f"frame_{count}.jpg"
                )
                label_path = (
                    params.dataset_dir / "labels" / f"frame_{count}.txt"
                )
                cv.imwrite(str(img_path), cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))
                self._imwrite_txt(label_path, "\n".join(yolo_lines))

            if out is not None:
                out.write(frame)

            count += 1

        cap.release()
        if out is not None:
            out.release()

        if params.dataset_dir:
            names = [
                label
                for label, idx in sorted(class_to_id.items(), key=lambda x: x[1])
            ]
            path = params.dataset_dir / "data.yaml"

            yaml_content = {
                "path": str(params.dataset_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(names),
                "names": names,
            }

            # Reuse klygo.io write_yaml helper for seamless integration
            write_yaml(path, yaml_content, overwrite=True, verbose=False)

        return annotated_images
