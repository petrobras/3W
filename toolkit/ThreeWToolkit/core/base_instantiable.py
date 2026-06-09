from typing import Any


class Instantiable:
    """
    Base class for builder/config pattern.
    Subclasses should set the 'target' attribute to the class to instantiate.
    The 'build' method instantiates the target with the config (self).
    'target' can be a type or a callable returning a type (e.g., from Field with default_factory).
    """

    _target: type["Any"]

    def build(self, *args, **kwargs) -> Any:
        return self._target(self, *args, **kwargs)
