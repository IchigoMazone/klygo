"""Shared deterministic fixtures for archive API tests."""

from pathlib import Path

import klygo.archive as archive
import klygo.files as files


def make_source(root: Path, name: str = "source") -> Path:
    source = root / name
    files.mkdir(source)
    files.save(source / "alpha.txt", "alpha", verbose=False)
    files.save(source / "beta.log", "beta", verbose=False)
    nested = source / "nested"
    files.mkdir(nested)
    files.save(nested / "data.json", {"value": 42}, verbose=False)
    return source


def make_archive(root: Path, name: str = "sample.zip", source_name: str = "source") -> Path:
    source = make_source(root, source_name)
    output = root / name
    archive.compress(source, output, overwrite=True, verbose=False)
    return output


def member_ending(names, suffix: str) -> str:
    return next(name for name in names if name.endswith(suffix))


__all__ = ["make_source", "make_archive", "member_ending"]
