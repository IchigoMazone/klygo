"""Convert YAML configuration to TOML."""

from tempfile import TemporaryDirectory

from klygo import config, files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = config.save(root / "a.yaml", {"seed": 7}, verbose=False)
    assert files.is_file(config.convert(source, root / "a.toml", verbose=False))
