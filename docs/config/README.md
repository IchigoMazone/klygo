# `klygo.config` Documentation

`klygo.config` provides hierarchical configuration management, dot-path nested access, immutable composition, environment variable overlays, structural comparison, schema validation, and a stateful object interface.

It also preserves the complete structures returned by `models.metadata()` and
`models.configure()`, including boolean runtime permissions and `None` values
in JSON, YAML, and TOML. See the dedicated
[model metadata configuration guide](model-metadata.md).

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/drive/1-aOofq_ZwLi00gRXnBLbJ4raupc6OZKm?usp=sharing)

---

## 1. Functional vs Stateful Interface

<details open>
<summary><b>Functional API vs. the stateful Config class</b> <i>(Click to expand or collapse)</i></summary>

`klygo.config` provides two complementary interfaces:

1. **Functional API (stateless and immutable)**:
   Functions such as `load`, `set`, `update`, `merge`, `flatten`, and `diff` return independent values without side effects on their inputs, making them suitable for functional data-transformation pipelines.
2. **Stateful `Config` class (path-bound and mutable)**:
   Use this interface when an application needs to retain a configuration instance, read or modify it in memory, and write it back to a file.

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (19 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Load configuration with `./` path expansion** | [`load()`](api/load.md) | `cfg = config.load("settings.yaml")` |
| **Save configuration to another format** | [`save()`](api/save.md) | `config.save("settings.toml", cfg, overwrite=True)` |
| **Convert configuration file format directly** | [`convert()`](api/convert.md) | `config.convert("settings.yaml", "settings.json")` |
| **Export configuration object or file** | [`export()`](api/export.md) | `config.export(cfg, "resolved.toml", overwrite=True)` |
| **Read nested value via dot notation** | [`get()`](api/get.md) | `batch = config.get(cfg, "model.batch", default=16)` |
| **Assign nested value (pure, non-mutating)** | [`set()`](api/set.md) | `cfg2 = config.set(cfg, "model.batch", 32)` |
| **Check if nested key exists (even if `None`)** | [`has()`](api/has.md) | `if config.has(cfg, "model.optimizer"): ...` |
| **Delete nested key in-place** | [`delete()`](api/delete.md) | `config.delete(cfg, "model.lr")` |
| **Recursively merge configurations left-to-right**| [`merge()`](api/merge.md) | `combined = config.merge(defaults, env_overrides, cli_flags)` |
| **Apply updates to a configuration** | [`update()`](api/update.md) | `cfg2 = config.update(cfg, {"model": {"epochs": 50}})` |
| **Flatten nested dictionary to dot paths** | [`flatten()`](api/flatten.md) | `flat = config.flatten(cfg)  # {'model.batch': 16}` |
| **Expand dot paths back to nested dictionary** | [`unflatten()`](api/unflatten.md) | `nested = config.unflatten({"model.batch": 16})` |
| **Overlay environment variables (`KLYGO_*`)** | [`from_env()`](api/from_env.md) | `cfg = config.from_env(cfg, prefix="KLYGO_", parse_values=True)` |
| **Inspect differences between two configs** | [`diff()`](api/diff.md) | `changes = config.diff(old_cfg, new_cfg)` |
| **List flattened or top-level keys/values/items** | [`keys()`](api/keys.md), [`values()`](api/values.md), [`items()`](api/items.md) | `config.keys(cfg, flat=True)` |
| **Validate required paths and data types** | [`validate()`](api/validate.md) | `config.validate(cfg, {"model.batch": int})` |
| **Get default built-in configuration template** | [`defaults()`](api/defaults.md) | `tmpl = config.defaults({"model": {"name": "yolov8m"}})` |
| **Create default config file on disk** | [`create()`](api/create.md) | `cfg = config.create("settings.yaml", overwrite=True)` |
| **Manage config via stateful class interface** | [`Config`](api/Config.md) | `manager = Config("settings.yaml"); manager.read()` |
| **Persist a model metadata schema exactly** | [`Config.create_default()`](api/Config.md) | `Config.create_default("model.toml", metadata=schema)` |
| **Reload configured model metadata** | [`load()`](api/load.md) | `model = models.load(config.load("model.yaml"))` |

</details>

---

## 3. Public APIs Reference (20 Functions, 1 Class)

<details>
<summary><b>File I/O and format conversion (4 functions)</b> <i>(Click to expand)</i></summary>

- [`load(path, verbose=True)`](api/load.md): Load configuration file into dot-accessible `Box` with `./` expansion.
- [`save(path, data, overwrite=False, ...)`](api/save.md): Save configuration mapping or `Box` to destination format.
- [`convert(source, target, overwrite=False, ...)`](api/convert.md): Format conversion between configuration files.
- [`export(source, target, overwrite=False, ...)`](api/export.md): Export configuration object or file to destination file.

</details>

<details>
<summary><b>Configuration creation and merging (4 functions)</b> <i>(Click to expand)</i></summary>

- [`defaults(default_data=None)`](api/defaults.md): Return a fresh copy of Klygo default configuration template.
- [`create(path, default_data=None, ...)`](api/create.md): Create and reload configuration file from defaults.
- [`merge(*configs, deep=True)`](api/merge.md): Merge multiple configurations left to right.
- [`update(config_data, updates, deep=True)`](api/update.md): Deep update configuration mapping without mutating input.

</details>

<details>
<summary><b>Nested value access and assignment with dot paths (4 functions)</b> <i>(Click to expand)</i></summary>

- [`get(config_data, key_path, default=None)`](api/get.md): Retrieve nested value by dot path (e.g. `"model.batch"`).
- [`set(config_data, key_path, value)`](api/set.md): Return copied configuration with nested value assigned.
- [`has(config_data, key_path)`](api/has.md): Distinguish missing key from key explicitly storing `None`.
- [`delete(config_data, key_path)`](api/delete.md): Delete nested key from mutable mapping in-place.

</details>

<details>
<summary><b>Structure inspection and transformation (6 functions)</b> <i>(Click to expand)</i></summary>

- [`keys(config_data, flat=False)`](api/keys.md): Return top-level or flattened dot-path keys.
- [`values(config_data, flat=False)`](api/values.md): Return top-level or flattened values.
- [`items(config_data, flat=False)`](api/items.md): Return top-level or flattened `(key, value)` tuples.
- [`diff(config1, config2)`](api/diff.md): Compare two configs or files returning added, removed, and modified values.
- [`flatten(config_data, sep='.')`](api/flatten.md): Flatten nested mappings into single-level dot-path mapping.
- [`unflatten(flat_dict, sep='.')`](api/unflatten.md): Reconstruct nested dictionary from dot-path mapping.

</details>

<details>
<summary><b>Environment, validation, and class interface (2 functions, 1 class)</b> <i>(Click to expand)</i></summary>

- [`from_env(config_data=None, prefix='KLYGO_', ...)`](api/from_env.md): Overlay environment variables onto configuration.
- [`validate(config_data, required_keys=None)`](api/validate.md): Validate required paths and schema type contracts.
- [`Config(config_path)`](api/Config.md): Stateful object-oriented configuration interface.

</details>

---

## 4. Model Metadata Workflow

```python
from klygo import Config, models

flags = models.flags(model=False, processor=True, post=True)
schema = models.metadata(
    flags,
    models.priority(flags, post=("threshold", "iou")),
)
configured = models.configure(
    "weights.pt",
    metadata=schema,
    implementation="my_package.CustomDetector",
    threshold=0.4,
)

manager = Config.create_default(
    "model.toml",
    metadata=configured,
    overwrite=True,
    verbose=False,
)
```

`metadata=` selects exact metadata mode, so general Klygo defaults are not
injected. The returned manager is already loaded. Full behavior, format notes,
runtime permissions, and examples are documented in
[Model metadata configuration](model-metadata.md).
