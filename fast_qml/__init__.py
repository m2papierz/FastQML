import importlib
from jax import config as j_cfg

# Default configuration set to cpu without jax boost
_device = 'cpu'
numpy = importlib.import_module("jax.numpy")
QUBIT_DEV = "default.qubit"


def update(device='CPU'):
    if device == 'cpu':
        new_np = importlib.import_module("pennylane.numpy")
        new_qubit_dev = "default.qubit"
    elif device == 'cpu.jax':
        j_cfg.update("jax_platform_name", "cpu")
        new_np = importlib.import_module("jax.numpy")
        new_qubit_dev = "default.qubit.jax"
    elif device == 'qpu':
        raise NotImplementedError
    else:
        raise ValueError()

    globals()['numpy'] = new_np
    globals()['QUBIT_DEV'] = new_qubit_dev


__all__ = ['numpy', 'update', 'QUBIT_DEV']
