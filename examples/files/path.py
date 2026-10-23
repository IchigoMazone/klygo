"""Executable example for klygo.files.path."""

from klygo import files

value = files.path("dataset/images")
print(value, type(value))


# Additional cases
home = files.path("~")
literal = files.path("~", expand_user=False)
assert home != literal and literal.name == "~"

# Custom filesystem inputs use __fspath__, not their string representation.
class DatasetPath:
    def __fspath__(self):
        return "dataset/images/cat.jpg"


native = files.path(DatasetPath())
assert native.name == "cat.jpg"
assert native.parent.as_posix() == "dataset/images"
