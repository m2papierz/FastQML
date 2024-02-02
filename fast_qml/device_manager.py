from enum import Enum

try:
    from jax import config as j_cfg
except ImportError:
    j_cfg = None


class QubitDevice(Enum):
    """
    Enumeration for different types of qubit devices.

    Attributes:
        CPU: Represents a default Pennylane CPU device interface.
        CPU_JAX: Represents a CPU device specifically for JAX interface.
    """
    CPU = 'cpu'
    CPU_JAX = 'cpu.jax'


class DeviceManager:
    """
    Manages the device settings for quantum computing operations.

    This class provides an interface to update and retrieve the current
    computational device being used, such as a CPU or a CPU configured for JAX.

    Attributes:
        device: The current device being used for computations.
    """
    def __init__(self):
        """
        Initializes the DeviceManager with a default device.
        """
        self.device: str = QubitDevice.CPU.value

    def update(self, device: str):
        """
        Updates the computational device.

        Validates the input device and updates the computational backend accordingly.
        If the device is set for JAX (CPU_JAX), it also updates JAX configuration.

        Args:
            device: The device to be set. Must be one of the values in QubitDevice.

        Raises:
            ValueError: If the provided device is not a valid QubitDevice.
            ImportError: If JAX is required but not installed.
        """
        if device not in [d.value for d in QubitDevice]:
            raise ValueError(
                f"Invalid device: '{device}'. "
                f"Valid devices are: {[d.value for d in QubitDevice]}"
            )

        if device == QubitDevice.CPU_JAX.value:
            if j_cfg is None:
                raise ImportError(
                    "JAX is not installed. Please install JAX to use this device."
                )
            j_cfg.update("jax_platform_name", "cpu")
            j_cfg.update("jax_enable_x64", True)

        self.device = device
