# `klygo.models`

Configure model metadata, select backends, and load wrappers. Import with `from klygo import models`. Concrete model backends may require optional packages or downloaded weights; these examples use local wrapper and metadata paths only.

## Base and detection

| API | Example |
| --- | --- |
| [`base`](api/base.md) | [Run](../../examples/models/base.py) |
| [`BaseModel`](api/BaseModel.md) | [Run](../../examples/models/BaseModel.py) |
| [`Detector`](api/Detector.md) | [Run](../../examples/models/Detector.py) |

## Loading and backend

| API | Example |
| --- | --- |
| [`load`](api/load.md) | [Run](../../examples/models/load.py) |
| [`configure`](api/configure.md) | [Run](../../examples/models/configure.py) |
| [`set_backend`](api/set_backend.md) | [Run](../../examples/models/set_backend.py) |
| [`get_backend`](api/get_backend.md) | [Run](../../examples/models/get_backend.py) |

## Metadata

| API | Example |
| --- | --- |
| [`metadata`](api/metadata.md) | [Run](../../examples/models/metadata.py) |
| [`priority`](api/priority.md) | [Run](../../examples/models/priority.py) |
| [`details`](api/details.md) | [Run](../../examples/models/details.py) |
| [`flags`](api/flags.md) | [Run](../../examples/models/flags.py) |

## Utilities and errors

| API | Example |
| --- | --- |
| [`utils`](api/utils.md) | [Run](../../examples/models/utils.py) |
| [`errors`](api/errors.md) | [Run](../../examples/models/errors.py) |
| [`suppress_warnings`](api/suppress_warnings.md) | [Run](../../examples/models/suppress_warnings.py) |
| [`suppress_ai_warnings`](api/suppress_ai_warnings.md) | [Run](../../examples/models/suppress_ai_warnings.py) |

Start with the [overview notebook](../../tutorials/models/Document_Models_Overview.ipynb), then [Base](../../tutorials/models/Document_Models_Base.ipynb), [Load](../../tutorials/models/Document_Models_Load.ipynb), [Metadata](../../tutorials/models/Document_Models_Metadata.ipynb), or [Utils](../../tutorials/models/Document_Models_Utils.ipynb).
