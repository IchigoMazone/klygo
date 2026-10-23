"""Executable examples of native pathlib integration with klygo.files."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

p = files.path("dataset/images/cat.jpg")
assert isinstance(p, Path)
assert p.name == "cat.jpg"
assert p.parent.as_posix() == "dataset/images"
assert files.path("~", expand_user=False) == Path("~")
assert files.path("~") == Path.home()

archive = files.path("backup.tar.gz")
assert archive.with_suffix(".zip").name == "backup.tar.zip"
assert files.with_extension(archive, "zip").name == "backup.zip"
assert files.compound_extension("report.v1.json") == ".v1.json"

with TemporaryDirectory() as directory:
    root = files.path(directory)
    note = root / "note.txt"
    note.write_text("Hello\n", encoding="utf-8")
    assert note.is_file()
    assert note.read_text(encoding="utf-8") == "Hello\n"

    renamed = note.with_name("renamed.txt")
    assert note.exists() and not renamed.exists()
    note.rename(renamed)
    assert renamed.exists() and not note.exists()

    config = root / "config.json"
    files.save(config, {"epochs": 10}, atomic=True, verbose=False)
    assert files.load(config, verbose=False) == {"epochs": 10}
    assert files.info(config)["size"] > 0
    assert files.is_within(config, root)
    assert not files.is_within(root / ".." / "outside.txt", root)

    link = root / "linked.txt"
    try:
        link.symlink_to(renamed.resolve())
    except OSError as error:
        if getattr(error, "winerror", None) != 1314 and error.errno not in (1, 13, 95):
            raise
        print("Symlink example skipped: this environment cannot create symlinks.")
    else:
        assert link.is_symlink() and link.is_file()
        assert link.readlink() == renamed.resolve()
        assert link.read_text(encoding="utf-8") == "Hello\n"
        renamed.unlink()
        assert not link.exists() and link.is_symlink()
        assert files.unique_path(link) != link
        files.remove(link, missing_ok=False)
        assert not link.is_symlink()
