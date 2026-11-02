"""Detect canonical formats from compound extensions."""

from klygo.archive.backend import detect_format


for filename in ("dataset.zip", "dataset.tgz", "dataset.tar.xz", "weights.gz"):
    print(filename, "->", detect_format(filename))

