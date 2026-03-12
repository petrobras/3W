from typing import Any, Type
from dataclasses import dataclass, field


class Instantiable:
    """
    Base class for builder/config pattern.
    Subclasses should set the 'target' attribute to the class to instantiate.
    The 'build' method instantiates the target with the config (self).
    'target' can be a type or a callable returning a type (e.g., from Field with default_factory).
    """

    target_: type["Any"]

    def build(self) -> Any:
        if isinstance(self.target_, type):
            target_type = self.target_
        elif callable(self.target_):
            result = self.target_()
            if not isinstance(result, type):
                raise TypeError("'target_' callable must return a type.")
            target_type = result
        else:
            raise TypeError("'target_' must be a type or a callable returning a type.")
        return target_type(self)
