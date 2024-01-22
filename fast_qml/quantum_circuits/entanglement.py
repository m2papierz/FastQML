import pennylane as qml
from typing import Any, Callable


def _linear(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    for i in range(n_qubits - 1):
        controlled_gate(wires=[i, i + 1])


def _circular(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    for i in range(n_qubits - 1):
        controlled_gate(wires=[i, i + 1])
    controlled_gate(wires=[n_qubits, 0])


def _full(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            controlled_gate(wires=[i, j])


class EntanglementGenerator:
    """
    Entangler class for generating quantum circuit entanglement patterns.

    Parameters:
    - n_qubits (int): Number of qubits in the quantum circuit.
    - c_gate (str): Controlled gate type ('CX', 'CY', or 'CZ').
    - entanglement (str): Entanglement pattern ('linear', 'circular', or 'full').
    """

    GATE_MAP = {
        'CX': qml.CNOT, 'CY': qml.CY, 'CZ': qml.CZ
    }

    ENTANGLEMENT_MAP = {
        'linear': _linear,
        'circular': _circular,
        'full': _full
    }

    def __init__(
            self,
            n_qubits: int,
            c_gate: str = 'CX',
            entanglement: str = 'linear'
    ):
        self._n_qubits = n_qubits
        self._entanglement = entanglement
        self._c_gate = self._get_gate(c_gate)

    def _get_gate(self, c_gate: str) -> Any:
        if c_gate in self.GATE_MAP:
            return self.GATE_MAP[c_gate]
        else:
            raise ValueError(
                "Invalid controlled gate type. "
                "Supported types are 'CX', 'CY', and 'CZ'."
            )

    def get_entanglement(
            self,
            entanglement: str
    ) -> Callable[[int, Any], None]:
        if entanglement in self.ENTANGLEMENT_MAP:
            return self.ENTANGLEMENT_MAP[entanglement]
        else:
            raise ValueError(
                "Invalid entanglement type."
                "Supported types are 'linear', 'circular', and 'full'."
            )

    def apply(self) -> None:
        self.get_entanglement(
            entanglement=self._entanglement
        )(self._n_qubits, self._c_gate)
