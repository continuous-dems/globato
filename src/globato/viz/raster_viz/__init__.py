"""Generic raster visualization and catalog interface (no producer dependencies)."""

from .catalog import ProductCatalog

__version__ = "0.13.11"
__all__ = ["Review", "ProductCatalog"]


def __getattr__(name):
    # Catalog publication does not need the rendering stack.
    if name == "Review":
        from .review import Review

        return Review
    raise AttributeError(name)
