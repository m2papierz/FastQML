# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from typing import Any, Callable, Union

import numpy as np
import pennylane as qml


def _linear(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    for i in range(n_qubits - 1):
        controlled_gate(wires=[i, i + 1])


def _reverse_linear(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    for i in range(n_qubits - 1, 0, -1):
        controlled_gate(wires=[i, i - 1])


def _circular(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    _linear(
        n_qubits=n_qubits, controlled_gate=controlled_gate)
    controlled_gate(wires=[n_qubits - 1, 0])


def _reverse_circular(
        n_qubits: int,
        controlled_gate: Any
) -> None:
    _reverse_linear(
        n_qubits=n_qubits, controlled_gate=controlled_gate)
    controlled_gate(wires=[0, n_qubits - 1])


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
        'linear_reverse': _reverse_linear,
        'circular': _circular,
        'reverse_circular': _reverse_circular,
        'full': _full
    }

    def __init__(
            self,
            n_qubits: int,
            c_gate: str = 'CX',
            entanglement: Union[str, list[list[int]]] = 'linear'
    ):
        self._n_qubits = n_qubits
        self._c_gate = self._get_gate(c_gate)
        self._entanglement = entanglement

    def _get_gate(self, c_gate: str) -> Any:
        if c_gate in self.GATE_MAP:
            return self.GATE_MAP[c_gate]
        else:
            raise ValueError(
                f"Invalid controlled gate type. "
                f"Supported types are {self.GATE_MAP.keys()}."
            )

    def _get_entanglement_function(
            self,
            entanglement: str
    ) -> Callable[[int, Any], None]:
        if entanglement in self.ENTANGLEMENT_MAP:
            return self.ENTANGLEMENT_MAP[entanglement]
        else:
            raise ValueError(
                f"Invalid entanglement type."
                f"Supported types are {self.ENTANGLEMENT_MAP.keys()}."
            )

    def _get_entanglement_scheme(
            self,
            entanglement: list[list[int]]
    ):
        if not all(isinstance(pair, list) and len(pair) == 2 for pair in entanglement):
            raise ValueError(
                "Entanglement scheme must be a list of qubit pairs."
            )

        qubits_needed = np.max(np.array(entanglement)) + 1
        if qubits_needed > self._n_qubits:
            raise ValueError(
                f"Given entanglement scheme needs {qubits_needed} qubits, "
                f"but circuit uses {self._n_qubits}."
            )

        for wires_pairs in entanglement:
            self._c_gate(wires=wires_pairs)

    def apply(self) -> None:
        if isinstance(self._entanglement, list):
            self._get_entanglement_scheme(
                entanglement=self._entanglement
            )
        else:
            self._get_entanglement_function(
                entanglement=self._entanglement
            )(self._n_qubits, self._c_gate)
