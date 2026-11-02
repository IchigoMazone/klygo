"""Builders for readable, validated Klygo model metadata."""

from typing import Any, Dict, Mapping, Sequence

from box import Box

from . import utils


def details(
    *,
    name: str,
    task: str,
    backend: str,
    library: str,
    **information: Any,
) -> Box:
    """Create the descriptive portion of model metadata.

    Parameters
    ----------
    name : str
        Human-readable model name.
    task : str
        Primary task performed by the model.
    backend : str
        Runtime or integration responsible for executing the model.
    library : str
        Upstream library that provides the model implementation.
    **information : object
        Additional serializable details such as ``version``, ``revision``,
        ``num_params``, ``author``, or ``license``. Values equal to ``None``
        are omitted.

    Returns
    -------
    box.Box
        Independent model details supporting mapping and attribute access.

    Raises
    ------
    TypeError
        If a required field is not a string.
    ValueError
        If a required field is empty or contains only whitespace.
    """
    required = {
        "name": name,
        "task": task,
        "backend": backend,
        "library": library,
    }
    for field, value in required.items():
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        if not value.strip():
            raise ValueError(f"{field} must not be empty")
        required[field] = value.strip()

    required.update(
        {str(field): value for field, value in information.items() if value is not None}
    )
    return Box(required)


def flags(*groups: str, **permissions: bool) -> Box:
    """Declare model parameter groups and predict-time permissions.

    Keyword declarations use ``True`` for groups that ``model.predict`` may
    override and ``False`` for groups locked after configuration. Positional
    names remain supported and are equivalent to setting every group to
    ``True``. The two forms cannot be mixed.
    """
    if groups and permissions:
        raise TypeError("flags accepts either positional groups or keyword permissions, not both")
    return Box(utils.normalize_flags(permissions or groups))


def priority(
    flags: Mapping[str, bool] | Sequence[str],
    **parameters: Any,
) -> Box:
    """Declare original parameter names that do not require group prefixes.

    Parameters are grouped by their destination. A name may belong to only one
    group, and every destination group must exist in ``flags``. Parameters not
    listed here remain available through ``<group>_<parameter>`` syntax.
    """
    normalized_flags = utils.normalize_flags(flags)
    normalized_groups = {name: {} for name in normalized_flags}
    return Box(utils.normalize_priority(normalized_groups, parameters))


def metadata(
    flags: Mapping[str, bool] | Sequence[str],
    priority: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> Box:
    """Combine model details, group flags, and priority aliases.

    This builder describes the parameter schema only. It does not initialize
    group mappings or parameter values; :func:`models.configure` performs that
    step. Model implementation classes belong to the loading registry and are
    intentionally not accepted here. The result always contains ``flags``,
    ``priority``, ``details``, and ``class``; omitted optional components and
    the loader-owned class field are ``None``.
    """
    normalized_flags = utils.normalize_flags(flags)
    normalized_details = None
    if details is not None:
        if not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")
        missing = tuple(
            field
            for field in ("name", "task", "backend", "library")
            if not isinstance(details.get(field), str) or not details[field].strip()
        )
        if missing:
            raise ValueError(f"details is missing required fields: {missing}")
        normalized_details = dict(details)

    result: Dict[str, Any] = {
        "flags": normalized_flags,
        "priority": None,
        "details": normalized_details,
        "class": None,
    }
    if priority is not None:
        result["priority"] = utils.normalize_priority(
            {name: {} for name in normalized_flags},
            priority,
        )
    return Box(result)


__all__ = ["details", "flags", "metadata", "priority"]
