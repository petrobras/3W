import functools
import inspect

from pydantic import BaseModel
from abc import ABC


class GeneralUtils(ABC):
    @staticmethod
    def validate_func_args_with_pydantic(schema: type[BaseModel]):
        """Decorator to validate function arguments using Pydantic.

        Args:
            schema (type[BaseModel]): Class derived from `pydantic.BaseModel`
                that defines the validation schema for the function arguments.

        Returns:
            Callable: Decorated function that performs validation before execution.

        Raises:
            ValueError: If argument validation fails, wraps the original
            Pydantic exception and includes the function name in the error message.
        """

        def decorator(func):
            sig = inspect.signature(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                validated = schema(**bound_args.arguments)

                return func(**validated.model_dump())

            return wrapper

        return decorator
