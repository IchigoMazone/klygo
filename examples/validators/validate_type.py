"""Validate a caller-supplied argument and display a useful error."""

from klygo import validators


source = "image.jpg"
validators.validate_type(source, str, "source")
print("Accepted source:", source)

try:
    validators.validate_type(42, (str, bytes), "source")
except TypeError as error:
    print("Invalid source:", error)
