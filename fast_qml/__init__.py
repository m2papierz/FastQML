from enum import Enum
from jax import config as j_cfg


class QubitDevice(Enum):
    CPU = 'cpu'
    CPU_JAX = 'cpu.jax'


DEVICE = QubitDevice.CPU


def update(device: str):
    if device not in [d.value for d in QubitDevice]:
        raise ValueError()

    if device == 'cpu.jax':
        j_cfg.update("jax_platform_name", "cpu")

    globals()['DEVICE'] = device


__all__ = ['update', 'DEVICE', 'QubitDevice']
