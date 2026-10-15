"""Shared filesystem input types; paths follow the host operating system."""

from os import PathLike
from typing import Union

PathInput = Union[str, PathLike[str]]
