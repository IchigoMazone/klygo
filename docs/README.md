# Klygo Documentation

Klygo provides Python utilities for archive operations, filesystem access, configuration, datasets, and object detection.

## Modules

- [`klygo.archive`](archive/README.md): Create, inspect, extract, validate, and modify ZIP, TAR, GZip, 7z, and RAR archives.
- [`klygo.archive.backend`](archive/backend/README.md): Backend interfaces, capabilities, and format-specific implementations.
- [`klygo.files`](files/README.md): Filesystem and path operations, streaming downloads, checksums, and structured-data I/O.
- [`klygo.config`](config/README.md): Hierarchical configuration, dot-path access, environment overlays, validation, and the `Config` class.
- [`klygo.datasets`](datasets/README.md): Load, inspect, partition, split, merge, remap, and repartition datasets.
- [`klygo.utils`](utils/README.md): Human-readable byte sizes and progress utilities.
- [`klygo.validators`](validators/README.md): Runtime input validation.
- [`klygo.visual`](visual/README.md): Image annotation, display, and dataset charts.
- [`klygo.outputs`](outputs/README.md): Structured object-detection results.
- [`klygo.media`](media/README.md): Lazy image and video loading, streaming, conversion, and output.
- [`klygo.postprocessing`](postprocessing/README.md): Composable detection-result transformations.
- [`klygo.models`](models/README.md): Model metadata, loading, and backend wrappers.
- [`klygo.preprocessing`](preprocessing/README.md): Image transforms, metrics, and coordinate restoration.

## Workflows

- [Real-image integration workflows](integration/README.md): Run multiple Klygo modules together and inspect isolated reports.

## Conventions

- Import public APIs from their package facade, for example `from klygo import files, archive, config`.
- Functional configuration operations return independent values unless an API explicitly documents mutation.
- File and archive paths accept strings and text-returning `os.PathLike[str]` objects.
- In archive, config, datasets, and media source code, use `klygo.files` for runtime path conversion, inspection, traversal, and filesystem changes; reserve `pathlib.Path` for type annotations. Codec backends use their format libraries to read and write archive contents.
- Pass `overwrite=True` when replacing existing output files where supported.
