import importlib
from jax import config as j_cfg

# Default parameters
_device = 'CPU'
numpy = importlib.import_module("jax.numpy")
QUBIT_DEV = "default.qubit"


def update(device='CPU'):
    if device == 'GPU':
        new_np = importlib.import_module("pennylane.numpy")
        new_qubit_dev = "lightning.qubit"
    else:
        j_cfg.update("jax_platform_name", "cpu")
        new_np = importlib.import_module("jax.numpy")
        new_qubit_dev = "default.qubit"

    globals()['numpy'] = new_np
    globals()['QUBIT_DEV'] = new_qubit_dev


__all__ = ['numpy', 'update', 'QUBIT_DEV']
