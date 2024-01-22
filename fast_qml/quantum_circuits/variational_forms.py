import numpy as np
import pennylane as qml

from typing import Any
from abc import abstractmethod
from pennylane import numpy as qnp

from fast_qml.quantum_circuits.entanglement import EntanglementGenerator


class VariationalForm:
    """
    Base class for quantum variational forms. Defines a template for creating quantum
    circuits with variational parameters.

    Parameters:
    - n_qubits (int): Number of qubits in the quantum circuit.
    - rotation_blocks (list[str]): List of rotation gate names. Supported values are 'RX', 'RY', and 'RZ'.
    - controlled_gate (str): Name of the controlled gate. Default is 'CX' (controlled-X gate).
    - entanglement (str): Type of entanglement to be applied. Default is 'linear'.
    - reps (int): Number of repetitions in the variational circuit. Default is 1.
    """

    GATE_MAP = {
        'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ
    }

    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            entanglement: str = 'linear',
            skip_last_rotation: bool = False,
            reps: int = 1
    ):
        self._n_qubits = n_qubits
        self._skip_last_rotation = skip_last_rotation
        self._reps = reps

        self._rotation_blocks = [
            self._validate_gate(block_gate)
            for block_gate in rotation_blocks or ['RZ']
        ]

        self._entanglement = EntanglementGenerator(
            n_qubits=n_qubits,
            c_gate=controlled_gate,
            entanglement=entanglement
        )

    @abstractmethod
    def _init_params(self) -> None:
        pass

    def _validate_gate(self, c_gate: str) -> Any:
        if c_gate in self.GATE_MAP:
            return self.GATE_MAP[c_gate]
        else:
            raise ValueError(
                "Invalid controlled gate type. "
                "Supported types are 'RX', 'RY', and 'RZ'."
            )

    @abstractmethod
    def circuit(self) -> None:
        pass

    def __call__(self) -> None:
        self.circuit()


class TwoLocal(VariationalForm):
    """
    Quantum variational form with a two-local structure.

    Parameters:
    - n_qubits (int): Number of qubits in the quantum circuit.
    - rotation_blocks (list[str]): List of rotation gate names. Supported values are 'RX', 'RY', and 'RZ'.
    - controlled_gate (str): Name of the controlled gate. Default is 'CX' (controlled-X gate).
    - entanglement (str): Type of entanglement to be applied. Default is 'linear'.
    - reps (int): Number of repetitions in the variational circuit. Default is 1.
    """

    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            entanglement: str = 'linear',
            skip_last_rotation: bool = False,
            reps: int = 1
    ):
        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            controlled_gate=controlled_gate,
            entanglement=entanglement,
            skip_last_rotation=skip_last_rotation,
            reps=reps
        )
        self._init_params()

    def _init_params(self) -> None:
        self._params = 0.01 * qnp.random.randn(
            self._reps + 1, self._n_qubits * len(self._rotation_blocks), 1,
            requires_grad=True
        )

    def circuit(self) -> None:
        def rotations(r_num):
            for i in range(self._n_qubits):
                for j, rot_ in enumerate(self._rotation_blocks):
                    rot_(self._params[r_num, j * self._n_qubits + i], wires=[i])

        for r in range(self._reps):
            rotations(r)
            self._entanglement.apply()

        if not self._skip_last_rotation:
            rotations(-1)

class EfficientSU2(TwoLocal):
    """
    Quantum variational form with an efficient SU(2) structure.

    Parameters:
    - n_qubits (int): Number of qubits in the quantum circuit.
    - rotation_blocks (list[str]): List of rotation gate names. If None, default is ['RY', 'RZ'].
    - controlled_gate (str): Name of the controlled gate. Default is 'CX' (controlled-X gate).
    - entanglement (str): Type of entanglement to be applied. Default is 'linear'.
    - reps (int): Number of repetitions in the variational circuit. Default is 1.
    """

    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            entanglement: str = 'linear',
            skip_last_rotation: bool = False,
            reps: int = 1
    ):
        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            controlled_gate=controlled_gate,
            entanglement=entanglement,
            skip_last_rotation=skip_last_rotation,
            reps=reps
        )
        if rotation_blocks is None:
            self._rotation_blocks = [qml.RY, qml.RZ]
        elif len(rotation_blocks) != 2:
            raise ValueError(
                "EfficientSU2 requires exactly 2 rotation blocks."
            )

        self._init_params()


class RealAmplitudes(TwoLocal):
    """
    Quantum variational form with real amplitudes. Inherits from the 'TwoLocal' class.

    Parameters:
    - n_qubits (int): Number of qubits in the quantum circuit.
    - entanglement (str): Type of entanglement to be applied. Default is 'linear'.
    - reps (int): Number of repetitions in the variational circuit. Default is 1.
    """

    def __init__(
            self,
            n_qubits: int,
            entanglement: str = 'linear',
            skip_last_rotation: bool = False,
            reps: int = 1
    ):
        super().__init__(
            n_qubits=n_qubits,
            entanglement=entanglement,
            skip_last_rotation=skip_last_rotation,
            reps=reps
        )

        self._rotation_blocks = [qml.RY]
        self._init_params()
