import importlib

_device = 'CPU'
numpy = importlib.import_module("jax.numpy")


def update(device='CPU'):
    if device == 'GPU':
        new_np = importlib.import_module("pennylane.numpy")
    else:
        new_np = importlib.import_module("jax.numpy")

    globals()['numpy'] = new_np


__all__ = ['numpy', 'update']
