"""Exchange mapping keys and values without mutating the source."""

from klygo import utils


classes = {"cat": 0, "dog": 1}
class_names = utils.invert(classes)
restored = utils.invert(class_names)

print(class_names)
print(restored)
