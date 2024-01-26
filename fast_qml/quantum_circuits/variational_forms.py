# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from abc import abstractmethod
from typing import Any, Union, Callable

import numpy as np
import pennylane as qml

from fast_qml.utils import validate_function_args
from fast_qml.quantum_circuits.entanglement import EntanglementGenerator


class VariationalForm:

    ROT_GATE_MAP = {
        'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ
    }

    def __init__(
            self,
            n_qubits: int,
            controlled_gate: str = None,
            reps: int = 1
    ):
        self._n_qubits = n_qubits
        self._controlled_gate = controlled_gate
        self._reps = reps

    def _validate_gate(
            self,
            c_gate: str
    ) -> Any:
        if c_gate not in self.ROT_GATE_MAP:
            raise ValueError(
                f"Invalid rotation gate type. "
                f"Supported types are {self.ROT_GATE_MAP.keys()}."
            )
        return self.ROT_GATE_MAP[c_gate]

    @property
    def params_num(self):
        return self._get_params_num()

    @abstractmethod
    def _get_params_num(self) -> int:
        pass

    @abstractmethod
    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        pass

    @abstractmethod
    def apply(
            self,
            params: np.ndarray
    ) -> None:
        pass


class Ansatz(VariationalForm):
    _expected_args = ['params']

    def __init__(
            self,
            n_qubits: int,
            parameters_num: int,
            variational_func: Callable = None,
            reps: int = 1
    ):

        self._parameters_num = parameters_num

        super().__init__(
            n_qubits=n_qubits,
            controlled_gate='CX',
            reps=reps
        )

        if not validate_function_args(variational_func, self._expected_args):
            raise ValueError(
                f"The variational_form function must "
                f"have the arguments: {self._expected_args}"
            )

        self._variational_function = variational_func

    def _get_params_num(self) -> int:
        return self._reps * self._parameters_num

    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        return self._variational_function(params)

    def apply(
            self,
            params: np.ndarray
    ) -> None:
        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )

        block_params_n = self._parameters_num
        for r in range(self._reps):
            params_subset = params[r * block_params_n: (r + 1) * block_params_n]
            self._variational_func(params=params_subset)


class TwoLocal(VariationalForm):
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            entanglement: Union[str, list[list[int]]] = 'linear',
            reps: int = 1
    ):
        super().__init__(
            n_qubits=n_qubits,
            controlled_gate=controlled_gate,
            reps=reps
        )

        self._entanglement = EntanglementGenerator(
            n_qubits=n_qubits,
            c_gate=controlled_gate,
            entanglement=entanglement
        )

        self._rotation_blocks = [
            self._validate_gate(block_gate)
            for block_gate in rotation_blocks or ['RY']
        ]

    def _get_params_num(self) -> int:
        rot_block_n = len(self._rotation_blocks)
        return self._reps * self._n_qubits * rot_block_n

    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        for j, rot_ in enumerate(self._rotation_blocks):
            for q in range(self._n_qubits):
                rot_(params[j * self._n_qubits + q], wires=[q])
        self._entanglement.apply()

    def apply(
            self,
            params: np.ndarray
    ) -> None:
        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )

        block_params_n = int(self._get_params_num() / self._reps)
        for r in range(self._reps):
            params_subset = params[r * block_params_n: (r + 1) * block_params_n]
            self._variational_func(params=params_subset)


class EfficientSU2(TwoLocal):
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            entanglement: str = 'linear',
            reps: int = 1
    ):
        if rotation_blocks is None:
            rotation_blocks = ['RY', 'RX']
        elif len(rotation_blocks) != 2:
            raise ValueError(
                "EfficientSU2 requires exactly 2 rotation blocks."
            )

        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            controlled_gate=controlled_gate,
            entanglement=entanglement,
            reps=reps
        )


class TreeTensor(VariationalForm):
    def __init__(
            self,
            n_qubits: int,
            controlled_gate: str = 'CX'
    ):
        if not (n_qubits & (n_qubits - 1)) == 0:
            raise ValueError(
                "TreeTensor ansatz requires the number of qubits "
                "to be a power of two."
            )

        reps_num = int(np.log2(n_qubits))

        super().__init__(
            n_qubits=n_qubits,
            controlled_gate=controlled_gate,
            reps=reps_num
        )

    def _get_params_num(self) -> int:
        return 2 * self._n_qubits - 1

    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        for i in range(self._n_qubits):
            qml.RY(params[i], wires=[i])

        n_qubits = self._n_qubits
        for r in range(1, self._reps + 1):
            for s in range(0, 2 ** (self._reps - r)):
                qml.CNOT(wires=[(s * 2 ** r), (s * 2 ** r) + (2 ** (r - 1))])
                qml.RY(params[n_qubits + s], wires=[(s * 2 ** r)])
            n_qubits += 2 ** (self._reps - r)

    def apply(
            self,
            params: np.ndarray
    ) -> None:
        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )
        self._variational_func(params=params)
