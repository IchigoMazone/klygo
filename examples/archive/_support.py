"""Reusable fixtures for executable archive examples."""

import klygo.archive as archive
import klygo.files as files


def make_source(root, name: str = "source"):
    base = files.path(root)
    source = files.mkdir(base / name)
    files.save(source / "alpha.txt", "alpha", verbose=False)
    files.save(source / "beta.log", "beta", verbose=False)
    nested = files.mkdir(source / "nested")
    files.save(nested / "data.json", {"value": 42}, verbose=False)
    return source


def make_archive(root, name: str = "sample.zip"):
    base = files.path(root)
    source = make_source(base)
    output = base / name
    archive.compress(source, output, overwrite=True, verbose=False)
    return output


def member_ending(names, suffix: str) -> str:
    return next(name for name in names if name.endswith(suffix))
