# Archive examples

This directory contains one executable example for every public name exported
by `klygo.archive`. Run examples as Python modules from the repository root:

```bash
python -m examples.archive.compress
python -m examples.archive.extract
python -m examples.archive.ArchiveFile
```

Module execution is important for `copy.py`: running a file from this directory
by path can make it shadow Python's standard-library `copy` module. Every
example uses temporary directories, performs assertions, and cleans up its
generated files automatically.

`_support.py` contains fixture helpers shared by the examples; it is not part of
the public archive API.
