import pennylane as qml

from typing import Any
from abc import abstractmethod

from jax import numpy as jnp
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

    def _validate_gate(
            self,
            c_gate: str
    ) -> Any:
        if c_gate in self.GATE_MAP:
            return self.GATE_MAP[c_gate]
        else:
            raise ValueError(
                "Invalid controlled gate type. "
                "Supported types are 'RX', 'RY', and 'RZ'."
            )

    @abstractmethod
    def get_params_num(self):
        pass

    def _validate_params_dims(
            self,
            params: jnp.ndarray
    ) -> None:
        if len(params) != self.get_params_num():
            raise ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.get_params_num()}, got {len(params)}."
            )

    @abstractmethod
    def apply(
            self,
            params: jnp.ndarray
    ) -> None:
        pass


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

    def get_params_num(self) -> int:
        if self._skip_last_rotation:
            layers_n = self._reps
        else:
            layers_n = self._reps + 1
        qubits_per_layer = self._n_qubits * len(self._rotation_blocks)
        return layers_n * qubits_per_layer

    def apply(
            self,
            params: jnp.ndarray
    ) -> None:
        def rotations(r_num):
            for i in range(self._n_qubits):
                for j, rot_ in enumerate(self._rotation_blocks):
                    rot_(params[r_num, j * self._n_qubits + i], wires=[i])

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
        if rotation_blocks is None:
            self._rotation_blocks = [qml.RY, qml.RZ]
        elif len(rotation_blocks) != 2:
            raise ValueError(
                "EfficientSU2 requires exactly 2 rotation blocks."
            )

        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            controlled_gate=controlled_gate,
            entanglement=entanglement,
            skip_last_rotation=skip_last_rotation,
            reps=reps
        )


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
        self._rotation_blocks = [qml.RY]

        super().__init__(
            n_qubits=n_qubits,
            entanglement=entanglement,
            skip_last_rotation=skip_last_rotation,
            reps=reps
        )


class TreeTensor(VariationalForm):
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            reps: int = 1
    ):
        if rotation_blocks is None:
            self._rotation_blocks = [qml.RY]

        if not (n_qubits & (n_qubits - 1)) == 0:
            raise ValueError(
                "TreeTensor ansatz requires the number of qubits "
                "to be a power of two."
            )

        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            controlled_gate=controlled_gate,
            reps=reps
        )

        self._reps = int(jnp.log2(n_qubits))

    def get_params_num(self) -> int:
        return 2 * self._n_qubits - 1

    def apply(
            self,
            params: jnp.ndarray
    ) -> None:
        self._validate_params_dims(params)

        for i in range(self._n_qubits):
            qml.RY(params[i], wires=[i])

        n_qubits = self._n_qubits
        for r in range(1, self._reps + 1):
            for s in range(0, 2 ** (self._reps - r)):
                qml.CNOT(wires=[(s * 2 ** r), (s * 2 ** r) + (2 ** (r - 1))])
                qml.RY(params[n_qubits + s], wires=[(s * 2 ** r)])
            n_qubits += 2 ** (self._reps - r)
