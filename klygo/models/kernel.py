import os
from pathlib import Path
from typing import Any, Union, List, Dict

import cv2 as cv
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

from klygo.validators.models import InitKernel, PredictKernel, DetectKernel
from klygo.io import write_yaml


class Kernel:
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
        """Initialize the Kernel object detection model.

        Parameters
        ----------
        model : str, optional
            The name of the model to load. Default is ``"Grounding-Dino/232M"``.

        Raises
        ------
        ValueError
            If the requested model is not supported or not implemented.
        """
        params = InitKernel(model=model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name_models = self.NAME_MODELS
        self.model_hugging_face = self.MODEL_HUGGING_FACE

        self.model = self._model(params.model)
        self.processor = self._processor(params.model)

        self.prompt = None
        self.return_tensors = None
        self.threshold = None
        self.text_threshold = None
        self.input_kwargs: Dict[str, Any] = {}
        self.output_kwargs: Dict[str, Any] = {}

    def _processor(self, model: str) -> Any:
        """Load the huggingface processor for the given model.

        Parameters
        ----------
        model : str
            The name of the model.

        Returns
        -------
        Any
            The processor object.
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
        """Load and transfer the huggingface model to the appropriate device.

        Parameters
        ----------
        model : str
            The name of the model.

        Returns
        -------
        Any
            The detection model.
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
        """Helper to write raw text content to a file."""
        with open(path, "w", encoding="utf-8") as file:
            file.write(data)

    def set_input_kwargs(self, **kwargs: Any) -> None:
        """Set extra keyword arguments passed directly to the model processor."""
        self.input_kwargs = kwargs

    def set_output_kwargs(self, **kwargs: Any) -> None:
        """Set extra keyword arguments passed directly to the post-processor."""
        self.output_kwargs = kwargs

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str = "apple.",
        return_tensors: str = "pt",
        threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Perform zero-shot object detection on one or more images.

        Parameters
        ----------
        images : PIL.Image.Image or list of PIL.Image.Image
            A single image or sequence of images to detect objects in.
        prompt : str, optional
            The text prompt mapping to target classes, e.g. ``"apple."`` or
            ``"car. dog."``. Default is ``"apple."``.
        return_tensors : str, optional
            The format of tensors to return. Default is ``"pt"``.
        threshold : float, optional
            The detection score threshold. Default is ``0.25``.
        text_threshold : float, optional
            The text detection alignment threshold. Default is ``0.25``.

        Returns
        -------
        list of dict
            A list of result dicts, each containing keys:
            ``"scores"``, ``"labels"``, and ``"boxes"`` (pixel coordinates).
        """
        params = PredictKernel(
            images=images,
            prompt=prompt,
            return_tensors=return_tensors,
            threshold=threshold,
            text_threshold=text_threshold,
        )

        self.prompt = params.prompt
        self.return_tensors = params.return_tensors
        self.threshold = params.threshold
        self.text_threshold = params.text_threshold
        target_size = [image.size[::-1] for image in params.images]

        inputs = self.processor(
            images=params.images,
            text=self.prompt,
            return_tensors=self.return_tensors,
            **self.input_kwargs,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_size,
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            **self.output_kwargs,
        )

    def detect(
        self,
        input_path: str,
        output_path: Union[str, None] = None,
        prompt: str = "apple.",
        return_tensors: str = "pt",
        threshold: float = 0.25,
        text_threshold: float = 0.25,
        desc: str = "Processing",
        save_frame_dir: Union[str, None] = None,
        save_yolo_dir: Union[str, None] = None,
    ) -> None:
        """Run object detection on frames of a video file.

        Optionally draws bounding boxes on frames, saves raw images, and/or
        exports dataset annotations in YOLO format.

        Parameters
        ----------
        input_path : str
            Path to the input video file.
        output_path : str or None, optional
            Destination path to export the annotated video.
        prompt : str, optional
            The detection text prompt. Default is ``"apple."``.
        return_tensors : str, optional
            Format of tensors to return. Default is ``"pt"``.
        threshold : float, optional
            Detection score threshold. Default is ``0.25``.
        text_threshold : float, optional
            Text alignment threshold. Default is ``0.25``.
        desc : str, optional
            Description label shown on the progress bar.
            Default is ``"Processing"``.
        save_frame_dir : str or None, optional
            Directory where annotated frame images will be saved.
        save_yolo_dir : str or None, optional
            Directory where YOLO-compatible image/label dataset will be saved.

        Raises
        ------
        FileNotFoundError
            If ``input_path`` does not exist or cannot be opened.
        """
        params = DetectKernel(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            return_tensors=return_tensors,
            threshold=threshold,
            text_threshold=text_threshold,
            desc=desc,
            save_frame_dir=save_frame_dir,
            save_yolo_dir=save_yolo_dir,
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
        out = None

        if params.output_path:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(
                str(params.output_path),
                fourcc,
                fps,
                (width, height),
            )

        if params.save_frame_dir:
            params.save_frame_dir.mkdir(parents=True, exist_ok=True)

        if params.save_yolo_dir:
            params.save_yolo_dir.mkdir(parents=True, exist_ok=True)
            (params.save_yolo_dir / "images").mkdir(parents=True, exist_ok=True)
            (params.save_yolo_dir / "labels").mkdir(parents=True, exist_ok=True)

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
                threshold=params.threshold,
                text_threshold=params.text_threshold,
            )

            yolo_lines = []
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"]
            ):
                x1, y1, x2, y2 = box.tolist()

                if params.output_path or params.save_frame_dir:
                    cv.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),
                        2,
                    )
                    cv.putText(
                        frame,
                        f"{label} {score:.2f}",
                        (int(x1), int(y1) - 5),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                if params.save_yolo_dir:
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

            if params.save_frame_dir:
                cv.imwrite(
                    str(params.save_frame_dir / f"frame_{count}.jpg"), frame
                )

            if params.save_yolo_dir:
                img_path = (
                    params.save_yolo_dir / "images" / f"frame_{count}.jpg"
                )
                label_path = (
                    params.save_yolo_dir / "labels" / f"frame_{count}.txt"
                )
                cv.imwrite(str(img_path), cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))
                self._imwrite_txt(label_path, "\n".join(yolo_lines))

            if out is not None:
                out.write(frame)

            count += 1

        cap.release()
        if out is not None:
            out.release()

        if params.save_yolo_dir:
            names = [
                label
                for label, idx in sorted(class_to_id.items(), key=lambda x: x[1])
            ]
            path = params.save_yolo_dir / "data.yaml"

            yaml_content = {
                "path": str(params.save_yolo_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(names),
                "names": names,
            }

            # Reuse klygo.io write_yaml helper for seamless integration
            write_yaml(path, yaml_content, overwrite=True, verbose=False)
