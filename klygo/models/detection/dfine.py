"""KerasHub D-FINE COCO detection with model-owned preprocessing and decoding."""
from collections import defaultdict
from contextlib import nullcontext
from math import isfinite
import numpy as np
from klygo import outputs
from klygo.models import utils
from klygo.models.detection.base import Detector
from klygo.models._frameworks import keras as execution
from klygo.models.runtime.keras import ensure_backend, _require_keras

__all__ = ["DFineDetect"]

_COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


def _restore_boxes(boxes, size, converter):
    """Invert the integer center crop/pad performed by Keras image.resize."""
    width, height = size
    target = converter.image_size
    if target is None:
        return boxes
    th, tw = target
    ox = oy = 0
    cw, ch = width, height
    if converter.crop_to_aspect_ratio:
        ch = max(min(height, int(width * th / tw)), 1)
        cw = max(min(width, int(height * tw / th)), 1)
        ox, oy = (width - cw) // 2, (height - ch) // 2
    elif converter.pad_to_aspect_ratio:
        ch = max(height, int(width * th / tw))
        cw = max(width, int(height * tw / th))
        ox, oy = -((cw - width) // 2), -((ch - height) // 2)
    return [[x1 * cw / tw + ox, y1 * ch / th + oy,
             x2 * cw / tw + ox, y2 * ch / th + oy] for x1, y1, x2, y2 in boxes]


class DFineDetect(Detector):
    """Run the KerasHub D-FINE COCO preset and return original-image boxes.

    Parameters
    ----------
    metadata : dict
        Registry metadata initialized by models.load. Model weights optionally
        select a local COCO preset directory. Device overrides require the Torch
        engine; other engines use their native placement.
    **kwargs : object
        Additional Detector initialization options.

    Notes
    -----
    Uses public preprocess_samples, model(training=False), and decode_predictions
    stages exactly once. Equal-size images share a batch; mixed-size batches are
    grouped by size and restored to source order. No extra resize or NMS is added.
    """
    def __init__(self, metadata, **kwargs):
        super().__init__(metadata=metadata, backend="Keras", **kwargs)
        ensure_backend()
        self._keras = _require_keras()
        try:
            import keras_hub
        except ImportError as exc:
            raise ImportError('D-FINE requires pip install "klygo[keras-hub]".') from exc
        config = dict(self.metadata.get("model", {}))
        device = config.get("device")
        if device is not None and self._keras.backend.backend() != "torch":
            raise ValueError("D-FINE device overrides require the Keras Torch engine")
        self._device_scope = device
        with self._scope():
            self.model = keras_hub.models.DFineObjectDetector.from_preset(config.get("weights") or self.model_id)
        if self.model.num_classes != 80:
            raise ValueError("D-FINE COCO presets must have exactly 80 classes")
        self.names = dict(enumerate(_COCO))

    def _scope(self):
        return self._keras.device(self._device_scope) if self._device_scope else nullcontext()

    def run_inference(self, inputs, **model_kwargs):
        """Run the native public preprocessing, forward, and decoder stages."""
        prepared = self.model.preprocess_samples(inputs)
        prepared, _, _ = self._keras.utils.unpack_x_y_sample_weight(prepared)
        raw = self.model(prepared, training=False, **model_kwargs)
        decoded = dict(self.model.decode_predictions(raw, prepared))
        decoded["_bounding_box_format"] = self.model.prediction_decoder.bounding_box_format
        return decoded

    def format_results(self, raw_outputs):
        """Normalize KerasHub decoded output to XYXY records."""
        return execution.format_results(raw_outputs)

    def forward(self, images, prompt=None):
        """Run D-FINE with original image geometry and optional COCO filtering."""
        allowed = set(utils.normalize_prompt(prompt)) if prompt else None
        if allowed and not allowed.issubset(self.names.values()):
            raise ValueError("D-FINE prompts must contain known COCO class names")
        post = dict(self.metadata.get("post", {}))
        threshold, iou = float(post.get("threshold", 0.25)), float(post.get("iou", 0.5))
        if not all(isfinite(value) and 0 <= value <= 1 for value in (threshold, iou)):
            raise ValueError("threshold and iou must be finite values between 0 and 1")
        batches = defaultdict(list)
        for index, image in enumerate(images):
            batches[image.size].append((index, image))
        formatted = [None] * len(images)
        decoder = self.model.prediction_decoder
        previous = decoder.confidence_threshold, decoder.iou_threshold
        try:
            decoder.confidence_threshold, decoder.iou_threshold = threshold, iou
            with self._scope():
                for size, batch in batches.items():
                    inputs = self._keras.ops.convert_to_tensor(np.stack([np.asarray(image) for _, image in batch]), dtype="float32")
                    rows = self.format_results(self.run_inference(inputs))
                    if len(rows) != len(batch):
                        raise ValueError("D-FINE must return one detection record per image")
                    for (index, _), row in zip(batch, rows):
                        row["boxes"] = _restore_boxes(row["boxes"], size, self.model.preprocessor.image_converter)
                        row["labels"] = [self.names[cid] for cid in row["class_ids"]]
                        selected = [i for i, score in enumerate(row["scores"])
                                    if score >= threshold and (allowed is None or row["labels"][i] in allowed)]
                        formatted[index] = {key: [row[key][i] for i in selected]
                                            for key in ("boxes", "scores", "labels", "class_ids")}
        finally:
            decoder.confidence_threshold, decoder.iou_threshold = previous
        return outputs.build_detections(images, formatted, names=self.names,
                                        metadata={"threshold": threshold, "iou": iou, "prompt": prompt})
