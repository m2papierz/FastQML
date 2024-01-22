import importlib

_device = 'CPU'
np = importlib.import_module("jax.numpy")


def update(device='CPU'):
    if device == 'GPU':
        new_np = importlib.import_module("pennylane.numpy")
    else:
        new_np = importlib.import_module("jax.numpy")

    globals()['np'] = new_np


__all__ = ['np', 'update']
