"""Copy a mapping while excluding one or more keys."""

from klygo import utils


settings = {"threshold": 0.4, "dtype": "float32", "device": "cpu"}
model_settings = utils.exclude_keys(settings, ["dtype", "device"])

print(model_settings)
print(settings)
