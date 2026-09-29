"""Load dot-accessible settings and model metadata."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import config, models


with TemporaryDirectory() as directory:
    root = Path(directory)

    settings_path = root / "settings.json"
    config.save(settings_path, {"model": {"batch": 16}}, verbose=False)
    assert config.load(settings_path, verbose=False).model.batch == 16

    flags = models.flags(model=False, processor=True, post=True)
    schema = models.metadata(flags)
    metadata_path = root / "model.toml"
    config.save(metadata_path, schema, verbose=False)

    restored = config.load(metadata_path, verbose=False)
    assert restored.flags.model is False
    assert restored.flags.processor is True
    assert restored["class"] is None
