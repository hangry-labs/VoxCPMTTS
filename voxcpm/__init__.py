from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import VoxCPM

__all__ = [
    "VoxCPM",
]


def __getattr__(name: str) -> Any:
    if name == "VoxCPM":
        from .core import VoxCPM

        return VoxCPM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
