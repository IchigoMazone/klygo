"""Persist model schemas and configured metadata in every supported format."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import Config, config, models


flags = models.flags(
    model=False,
    processor=True,
    post=True,
)

schema = models.metadata(
    flags=flags,
    priority=models.priority(
        flags,
        model=("device",),
        processor=("size",),
        post=("threshold", "iou"),
    ),
    details=models.details(
        name="Example detector",
        task="Object-Detection",
        backend="Custom",
        library="custom",
    ),
)

configured = models.configure(
    "weights.pt",
    metadata=schema,
    implementation="my_package.detectors.CustomDetector",
    device="cuda",
    size=640,
    threshold=0.4,
    post_iou=0.7,
)

with TemporaryDirectory() as directory:
    root = Path(directory)

    for suffix in ("json", "yaml", "toml"):
        schema_path = root / f"schema.{suffix}"
        manager = Config.create_default(
            schema_path,
            metadata=schema,
            verbose=False,
        )
        assert manager.get("flags.model") is False
        assert manager.get("flags.processor") is True
        assert manager.get("class") is None

        configured_path = root / f"configured.{suffix}"
        config.save(configured_path, configured, verbose=False)
        restored = config.load(configured_path, verbose=False)

        assert restored.model_id == "weights.pt"
        assert restored.model.device == "cuda"
        assert restored.processor.size == 640
        assert restored.post.threshold == 0.4
        assert restored.post.iou == 0.7
        assert restored.flags.model is False
