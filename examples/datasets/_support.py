"""Shared fixtures used by executable dataset examples."""

from typing import Any

from PIL import Image

from klygo import config, files, media
from klygo.outputs.detect import Box, Detection, Detections


def make_dataset(
    root: Any,
    names: tuple[str, ...] = ("apple", "banana"),
    prefix: str = "sample",
) -> Any:
    """Create a tiny valid YOLO dataset for examples."""
    for index, class_id in enumerate((0, 1, 0, 1)):
        split = "train" if index < 2 else "val"
        media.save(
            root / "images" / split / f"{prefix}_{index}.jpg",
            Image.new("RGB", (32, 24), "white"),
            verbose=False,
        )
        files.save(
            root / "labels" / split / f"{prefix}_{index}.txt",
            [f"{class_id} 0.5 0.5 0.5 0.5"],
            overwrite=True,
            verbose=False,
        )
    config.save(
        root / "data.yaml",
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": list(names),
        },
        overwrite=True,
        verbose=False,
    )
    return root


class ExampleDetector:
    """Return one full-image detection for every input image."""

    def __call__(self, images, prompt=None, **kwargs):
        detections = []
        for image in images:
            width, height = image.size
            detections.append(
                Detection(
                    source_image=image,
                    objects=[
                        Box(
                            id=0,
                            label=prompt[0],
                            score=0.9,
                            box=[0, 0, width, height],
                            parent_image=image,
                        )
                    ],
                )
            )
        return Detections(detections, source_type="images")
