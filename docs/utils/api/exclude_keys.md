# `utils.exclude_keys`

```python
utils.exclude_keys(values, keys)
```

Return a shallow dictionary copy without selected keys. Pass one key as a
string or multiple keys as a list of strings. The source mapping is not
modified. Unsupported key containers and non-string list items raise
`TypeError`.

```python
from klygo import utils

settings = {"threshold": 0.4, "dtype": "float32", "device": "cpu"}
model_settings = utils.exclude_keys(settings, ["dtype", "device"])

print(model_settings)
# {'threshold': 0.4}
```

See the [runnable example](../../../examples/utils/exclude_keys.py) and the
[`klygo.utils` index](../README.md).
