"""Save ordinary configuration and model metadata by destination extension."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import config, models


with TemporaryDirectory() as directory:
    root = Path(directory)

    settings_path = root / "settings.yaml"
    assert config.save(
        settings_path,
        {"debug": False},
        verbose=False,
    ) == settings_path

    flags = models.flags(model=False, processor=True, post=True)
    schema = models.metadata(flags)
    metadata_path = root / "model.toml"
    config.save(metadata_path, schema, verbose=False)

    restored = config.load(metadata_path, verbose=False)
    assert restored.to_dict() == schema.to_dict()
