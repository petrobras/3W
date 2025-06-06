import functools

from pydantic import BaseModel
from abc import ABC

class GeneralUtils(ABC):

    @staticmethod
    def validate_func_args_with_pydantic(schema: type[BaseModel]):
        """Decorador para validar os argumentos da função usando Pydantic.

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
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                validated = schema(*args, **kwargs)
                
                return func(**validated.model_dump())
            return wrapper
        return decorator