# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

import warnings
from abc import abstractmethod
from typing import Any, Union, Callable

import numpy as np
import pennylane as qml

from fast_qml.utils import validate_function_args
from fast_qml.quantum_circuits.entanglement import EntanglementGenerator


class VariationalForm:

    _expected_args = ['params']

    GATE_MAP = {
        'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ
    }

    def __init__(
            self,
            n_qubits: int,
            variational_func: Callable = None,
            entanglement: Union[str, list[int]] = None,
            rotation_blocks: list[str] = None,
            controlled_gate: str = None,
            uniform_structure: bool = True,
            reps: int = 1
    ):
        self._n_qubits = n_qubits
        self._reps = reps

        if variational_func is not None:
            if entanglement is not None or controlled_gate is not None:
                warnings.warn(
                    message="If variational_func is given, entanglement "
                            "and controlled_gate will be ignored.",
                    category=UserWarning
                )

            if not validate_function_args(variational_func, self._expected_args):
                raise ValueError(
                    f"The variational_form function must "
                    f"have the arguments: {self._expected_args}"
                )
            self._variational_func = variational_func
        else:
            self._entanglement = EntanglementGenerator(
                n_qubits=n_qubits,
                c_gate=controlled_gate,
                entanglement=entanglement
            )

            self._variational_func = self._set_variational_func

        self._rotation_blocks = [
            self._validate_gate(block_gate)
            for block_gate in rotation_blocks or ['RY']
        ]

        if uniform_structure:
            self._func_params_n = self._get_func_params_num()

    @property
    def params_num(self) -> int:
        return self._reps * self._func_params_n

    def _get_func_params_num(self) -> int:
        rot_block_n = len(self._rotation_blocks)
        aux_params_num = self._n_qubits * rot_block_n

        with qml.tape.QuantumTape() as tape:
            self._variational_func(np.zeros(shape=aux_params_num))

        return tape.num_params

    def _validate_gate(
            self,
            c_gate: str
    ) -> Any:
        if c_gate not in self.GATE_MAP:
            raise ValueError(
                "Invalid rotation gate type. "
                "Supported types are 'RX', 'RY', and 'RZ'."
            )
        return self.GATE_MAP[c_gate]

    @abstractmethod
    def _set_variational_func(
            self,
            params: np.ndarray
    ) -> Callable:
        pass

    def apply(
            self,
            params: np.ndarray
    ) -> None:
        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )

        qubits_n = self._func_params_n
        for r in range(self._reps):
            params_subset = params[r * qubits_n: (r + 1) * qubits_n]
            self._variational_func(params=params_subset)


class TwoLocal(VariationalForm):
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CX',
            entanglement: str = 'linear',
            reps: int = 1
    ):
        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=rotation_blocks,
            controlled_gate=controlled_gate,
            entanglement=entanglement,
            reps=reps
        )

    def _set_variational_func(
            self,
            params: np.ndarray
    ) -> None:
        def variational_func():
            for j, rot_ in enumerate(self._rotation_blocks):
                for q in range(self._n_qubits):
                    rot_(params[j * self._n_qubits + q], wires=[q])
            self._entanglement.apply()
        return variational_func()


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

        super().__init__(
            n_qubits=n_qubits,
            rotation_blocks=None,
            controlled_gate=controlled_gate,
            reps=1,
            uniform_structure=False
        )

        self._reps = int(np.log2(n_qubits))
        self._func_params_n = self.params_num

    @property
    def params_num(self) -> int:
        return 2 * self._n_qubits - 1

    def _set_variational_func(
            self,
            params: np.ndarray
    ) -> None:
        def variational_func():
            for i in range(self._n_qubits):
                qml.RY(params[i], wires=[i])

            n_qubits = self._n_qubits
            for r in range(1, self._reps + 1):
                for s in range(0, 2 ** (self._reps - r)):
                    qml.CNOT(wires=[(s * 2 ** r), (s * 2 ** r) + (2 ** (r - 1))])
                    qml.RY(params[n_qubits + s], wires=[(s * 2 ** r)])
                n_qubits += 2 ** (self._reps - r)

        return variational_func()

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
