# `utils.invert`

```python
utils.invert(values)
```

Return a new dictionary with every key and value exchanged. Values must be
unique and hashable so calling the function again restores the original
mapping. The source mapping is not modified.

```python
from klygo import utils

classes = {"cat": 0, "dog": 1}
class_names = utils.invert(classes)
restored = utils.invert(class_names)

print(class_names)  # {0: 'cat', 1: 'dog'}
print(restored)     # {'cat': 0, 'dog': 1}
```

Duplicate values raise `ValueError`; unhashable values and non-mapping inputs
raise `TypeError`.

See the [runnable example](../../../examples/utils/invert.py) and the
[`klygo.utils` index](../README.md).
