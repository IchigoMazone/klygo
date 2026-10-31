"""Build a small reusable detection dataset for mining examples."""

import numpy as np
from PIL import Image

from klygo import files, media, outputs


def sample_data() -> outputs.Detections:
    """Return four annotated images stored under the artifacts directory."""
    root = files.join("artifacts", "examples", "mining", "images")
    files.mkdir(root)
    names = {0: "car", 1: "person", 2: "bike"}
    labels = [("car",), ("car", "person"), ("person",), ("bike",)]
    frames = []
    for index, frame_labels in enumerate(labels):
        path = files.join(root, f"sample_{index}.png")
        pixels = np.full((64, 96, 3), 40 + index * 50, dtype=np.uint8)
        pixels[index * 8:index * 8 + 8] = 255
        media.save(path, Image.fromarray(pixels), overwrite=True, verbose=False)
        boxes = [outputs.Box([5 + offset * 10, 8, 35 + offset * 10, 42], label=label,
                             class_id={value: key for key, value in names.items()}[label])
                 for offset, label in enumerate(frame_labels)]
        frames.append(outputs.Detection(boxes, source_id=str(path), source_path=str(path),
                                        image_size=(96, 64), names=names))
    return outputs.Detections(frames, names=names)
