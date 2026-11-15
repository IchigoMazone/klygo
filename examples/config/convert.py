"""Convert YAML configuration to TOML."""

from tempfile import TemporaryDirectory

from klygo import config, files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "a.yaml"
    target = root / "a.toml"
    config.save(source, {"seed": 7}, verbose=False)
    config.convert(source, target, verbose=False)
    assert files.is_file(target)
