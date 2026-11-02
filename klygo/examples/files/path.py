"""Executable example for klygo.files.path."""

from klygo import files

value = files.path("dataset/images")
print(value, type(value))


# Additional cases
home = files.path("~")
literal = files.path("~", expand_user=False)
assert home != literal and literal.name == "~"

