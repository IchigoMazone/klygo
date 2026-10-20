"""Convert YAML configuration to TOML."""
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config
with TemporaryDirectory() as directory:
    root = Path(directory); source = root / "a.yaml"
    config.save(source, {"seed": 7}, verbose=False)
    assert config.convert(source, root / "a.toml", verbose=False).is_file()
