"""Use Config for general settings and exact model metadata."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import Config, models


with TemporaryDirectory() as directory:
    root = Path(directory)

    settings = Config.create_default(root / "settings.yaml", verbose=False)
    settings.set("model.batch", 32)
    assert settings.get("model.batch") == 32

    flags = models.flags(model=False, processor=True, post=True)
    schema = models.metadata(flags)
    metadata = Config.create_default(
        root / "model.toml",
        metadata=schema,
        verbose=False,
    )

    assert metadata.get("flags.model") is False
    assert metadata.get("flags.post") is True
    assert metadata.get("class") is None
