"""Executable examples for klygo.utils.human_size."""

from klygo import utils


for value in (0, 1024, 1024 ** 2, 1024 ** 3):
    print(value, "->", utils.human_size(value))

assert utils.human_size(1024, decimal_places=1) == "1.0 KB"
try:
    utils.human_size(-1)
except ValueError:
    print("negative sizes are rejected")
