import inspect
from typing import Callable


def validate_function_args(
        func: Callable,
        expected_args: list[str]
) -> bool:
    """

    Args:
        func:
        expected_args:

    Returns:

    """
    func_signature = inspect.signature(func)
    func_args = list(func_signature.parameters.keys())
    return set(expected_args) <= set(func_args)
