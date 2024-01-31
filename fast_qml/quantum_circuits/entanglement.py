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
    A class that creates entanglement patterns for quantum circuits using controlled gates. It supports
    predefined entanglement patterns such as linear, circular, full, and their reverse variants. Additionally,
    it allows for custom entanglement schemes.

    Args:
        n_qubits: The number of qubits in the quantum circuit.
        c_gate: The type of controlled gate to use. Defaults to 'CNOT'.
        entanglement: The entanglement pattern.Can be a string for predefined patterns or a list of qubit
        pairs for custom entanglement. Defaults to 'linear'.

    Attributes:
        ENTANGLEMENT_FUNCTIONS (dict): A mapping from entanglement pattern names to their
            corresponding methods for applying these patterns.
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
        """
        Retrieves the controlled gate function based on the specified gate name.

        Args:
            c_gate: The name of the controlled gate.

        Returns:
            The controlled gate function.

        Raises:
            ValueError: If the specified gate is not a valid Pennylane controlled gate or if
                the gate attribute does not exist.
        """
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
        """
        Applies a linear entanglement pattern to the quantum circuit.

        Args:
            n_qubits: The number of qubits in the circuit.
            controlled_gate: The controlled gate function to apply.
            reverse : If True, applies the entanglement in reverse order. Defaults to False.
        """
        qubit_range = range(n_qubits - 1) if not reverse else range(n_qubits - 1, 0, -1)
        for i in qubit_range:
            controlled_gate(wires=[i, i + (1 if not reverse else -1)])

    @staticmethod
    def _apply_circular(
            n_qubits: int,
            controlled_gate: Any,
            reverse: bool = False
    ) -> None:
        """
        Applies a circular entanglement pattern to the quantum circuit.

        Args:
            n_qubits: The number of qubits in the circuit.
            controlled_gate: The controlled gate function to apply.
            reverse: If True, applies the entanglement in reverse order. Defaults to False.
        """
        Entangler._apply_linear(n_qubits, controlled_gate, reverse)
        controlled_gate(wires=[n_qubits - 1, 0] if not reverse else [0, n_qubits - 1])

    @staticmethod
    def _apply_full(
            n_qubits: int,
            controlled_gate: Any
    ) -> None:
        """
       Applies a full entanglement pattern to the quantum circuit.

       Args:
           n_qubits: The number of qubits in the circuit.
           controlled_gate: The controlled gate function to apply.
       """
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                controlled_gate(wires=[i, j])

    def _apply_custom(
            self,
            entanglement_scheme: List[List[int]]
    ) -> None:
        """
        Applies a custom entanglement pattern to the quantum circuit.

        Args:
            entanglement_scheme: A list of qubit pairs specifying the custom entanglement pattern.

        Raises:
            ValueError: If the entanglement scheme is invalid or contains qubit indices
                outside the range of available qubits.
        """
        for pair in entanglement_scheme:
            if len(pair) != 2 or not all(0 <= qubit < self._n_qubits for qubit in pair):
                raise ValueError(
                    f"Custom entanglement scheme must be a list of valid qubit pairs "
                    f"with qubit indices in the range 0 to {self._n_qubits - 1}."
                )
            self._c_gate(wires=pair)

    def apply(self) -> None:
        """
        Applies the specified entanglement pattern to the quantum circuit.

        Raises:
            ValueError: If the entanglement type is not recognized or supported.
        """
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
