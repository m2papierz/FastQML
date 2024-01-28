import inspect
from typing import Callable


def validate_function_args(
        func: Callable,
        expected_args: list[str]
) -> bool:
    """
    Validates whether the given function's signature matches the expected arguments.

    Args:
        func: The target function to validate.
        expected_args: A list of expected argument names.

    Returns:
        True if the function's signature contains all the expected arguments, False otherwise.

    Example:
        >>> def example_function(param1, param2):
        ...     pass
        >>> validate_function_args(example_function, ['param1', 'param2'])
        True
        >>> validate_function_args(example_function, ['param1', 'param2', 'param3'])
        False
        >>> validate_function_args(example_function, ['param1'])
        False
    """
    func_signature = inspect.signature(func)
    func_args = list(func_signature.parameters.keys())
    return set(expected_args) == set(func_args)
