from typing import Any, Optional

from tqdm.auto import tqdm


class ProgressBar:
    """Context-managed progress indicator shared by Klygo modules.

    The wrapper provides one configuration surface for terminal and notebook
    progress output. When ``verbose=False``, no ``tqdm`` object is created and
    calls to :meth:`update` and :meth:`close` become safe no-ops.

    Parameters
    ----------
    total : int or None
        Expected number of updates. ``None`` or ``0`` uses a display total of
        one because ``tqdm`` requires a usable fallback for this wrapper.
    desc : str
        Text displayed before the progress bar.
    unit : str, default='file'
        Unit label such as ``file``, ``frame``, or ``byte``.
    verbose : bool, default=True
        Create and display the underlying progress bar when enabled.
    colour : str, default='cyan'
        Color name forwarded to ``tqdm``.
    unit_scale : bool, default=True
        Allow ``tqdm`` to abbreviate large values.
    unit_divisor : int, default=1000
        Scaling divisor, commonly 1000 for item counts or 1024 for bytes.

    Attributes
    ----------
    verbose : bool
        Whether progress display is enabled.
    bar : tqdm or None
        Active underlying progress object, or ``None`` when disabled or closed.

    Examples
    --------
    >>> from klygo import utils
    >>> with utils.ProgressBar(2, "Loading", verbose=False) as progress:
    ...     progress.update()
    ...     progress.update()
    """

    def __init__(
        self,
        total: Optional[int],
        desc: str,
        unit: str = "file",
        verbose: bool = True,
        colour: str = "cyan",
        unit_scale: bool = True,
        unit_divisor: int = 1000,
    ) -> None:
        self.verbose = verbose
        self.bar: Optional[Any] = None

        if verbose:
            self.bar = tqdm(
                total=total or 1,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=unit_divisor,
                colour=colour,
                ascii=" █",
                leave=True,
            )
    def update(self, n: int = 1) -> None:
        """Advance the progress counter by ``n`` units when enabled."""
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        """Close and release the underlying progress object when present."""
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def create_progress_bar(
    total: Optional[int],
    desc: str,
    unit: str = "file",
    verbose: bool = True,
    colour: str = "cyan",
    unit_scale: bool = False,
    unit_divisor: int = 1024,
) -> ProgressBar:
    """Create a configured :class:`ProgressBar`.

    This factory is useful when callers prefer function-based construction. Its
    defaults favor byte-oriented or explicitly controlled scaling, whereas the
    class constructor enables unit scaling by default.

    Parameters
    ----------
    total : int or None
        Expected number of updates.
    desc : str
        Text displayed before the progress bar.
    unit : str, default='file'
        Unit label displayed beside the counter.
    verbose : bool, default=True
        Create and display the underlying progress bar when enabled.
    colour : str, default='cyan'
        Color name forwarded to ``tqdm``.
    unit_scale : bool, default=False
        Allow ``tqdm`` to abbreviate large values.
    unit_divisor : int, default=1024
        Scaling divisor used by ``tqdm``.

    Returns
    -------
    ProgressBar
        Configured progress wrapper. The caller should close it or use it as a
        context manager.

    Examples
    --------
    >>> from klygo import utils
    >>> progress = utils.create_progress_bar(1, "Saving", verbose=False)
    >>> progress.update()
    >>> progress.close()
    """
    return ProgressBar(
        total=total,
        desc=desc,
        unit=unit,
        verbose=verbose,
        colour=colour,
        unit_scale=unit_scale,
        unit_divisor=unit_divisor,
    )
