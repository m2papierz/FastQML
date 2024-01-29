# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

"""
Module providing functionalities for generating quantum circuit entanglement patterns.
"""

from typing import Any, List, Union
import pennylane as qml
from pennylane.ops.op_math.controlled import ControlledOp


class Entangler:
    """
    This class generates various entanglement patterns for quantum circuits.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        c_gate: Controlled gate type. Defaults to 'CNOT'.
        entanglement: Entanglement pattern or custom pattern. Defaults to 'linear'.

    Attributes:
        ENTANGLEMENT_FUNCTIONS (dict): Mapping of entanglement pattern names to their corresponding functions.
    """

    ENTANGLEMENT_FUNCTIONS = {
        'linear': lambda n, gate: Entangler._apply_linear(n, gate),
        'reverse_linear': lambda n, gate: Entangler._apply_linear(n, gate, reverse=True),
        'circular': lambda n, gate: Entangler._apply_circular(n, gate),
        'reverse_circular': lambda n, gate: Entangler._apply_circular(n, gate, reverse=True),
        'full': lambda n, gate: Entangler._apply_full(n, gate)
    }

    def __init__(
            self,
            n_qubits: int,
            c_gate: str = 'CNOT',
            entanglement: Union[str, List[List[int]]] = 'linear'
    ):
        self._n_qubits = n_qubits
        self._c_gate = self._get_controlled_gate(c_gate)
        self._entanglement = entanglement

    @staticmethod
    def _get_controlled_gate(
            c_gate: str
    ) -> Any:
        try:
            gate_function = getattr(qml, c_gate)
            if not issubclass(gate_function, ControlledOp) and c_gate != 'CNOT':
                raise ValueError(
                    f"Given gate {c_gate} is not a valid Pennylane controlled gate."
                )
            return gate_function
        except AttributeError:
            raise ValueError(
                f"Invalid controlled gate type: {c_gate}."
            )

    @staticmethod
    def _apply_linear(
            n_qubits: int,
            controlled_gate: Any,
            reverse: bool = False
    ) -> None:
        qubit_range = range(n_qubits - 1) if not reverse else range(n_qubits - 1, 0, -1)
        for i in qubit_range:
            controlled_gate(wires=[i, i + (1 if not reverse else -1)])

    @staticmethod
    def _apply_circular(
            n_qubits: int,
            controlled_gate: Any,
            reverse: bool = False
    ) -> None:
        Entangler._apply_linear(n_qubits, controlled_gate, reverse)
        controlled_gate(wires=[n_qubits - 1, 0] if not reverse else [0, n_qubits - 1])

    @staticmethod
    def _apply_full(
            n_qubits: int,
            controlled_gate: Any
    ) -> None:
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                controlled_gate(wires=[i, j])

    def _apply_custom(
            self,
            entanglement_scheme: List[List[int]]
    ) -> None:
        for pair in entanglement_scheme:
            if len(pair) != 2 or not all(0 <= qubit < self._n_qubits for qubit in pair):
                raise ValueError(
                    f"Custom entanglement scheme must be a list of valid qubit pairs "
                    f"with qubit indices in the range 0 to {self._n_qubits - 1}."
                )
            self._c_gate(wires=pair)

    def apply(self) -> None:
        if isinstance(self._entanglement, list):
            self._apply_custom(self._entanglement)
        else:
            entanglement_func = self.ENTANGLEMENT_FUNCTIONS.get(self._entanglement)
            if not entanglement_func:
                raise ValueError(
                    f"Invalid entanglement type. "
                    f"Supported types are {list(self.ENTANGLEMENT_FUNCTIONS.keys())}."
                )
            entanglement_func(self._n_qubits, self._c_gate)
