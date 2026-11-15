"""Segmentation pipelines with lazy compatibility exports for TextTiling."""

from .texttiling import __all__


def __getattr__(name: str):
    if name in __all__:
        from . import texttiling

        return getattr(texttiling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
