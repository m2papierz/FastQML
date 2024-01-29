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
Module providing variational forms.
"""

from abc import abstractmethod
from typing import Any, Union, Callable

import numpy as np
import pennylane as qml

from fast_qml.utils import validate_function_args
from fast_qml.quantum_circuits.entanglement import Entangler


class VariationalForm:
    """
    Abstract base class for quantum variational forms.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        controlled_gate: Controlled gate type.
        reps: Number of repetitions.

    Attributes:
        ROT_GATE_MAP : Mapping of rotation gate types to PennyLane operations.
    """

    ROT_GATE_MAP = {
        'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ
    }

    def __init__(
            self,
            n_qubits: int,
            controlled_gate: str,
            reps: int = 1
    ):
        self._n_qubits = n_qubits
        self._controlled_gate = controlled_gate
        self._reps = reps

    def _validate_gate(
            self,
            c_gate: str
    ) -> Any:
        """
        Validates the rotation gate type.

        Args:
            c_gate: Rotation gate type.
        """
        if c_gate not in self.ROT_GATE_MAP:
            raise ValueError(
                f"Invalid rotation gate type. "
                f"Supported types are {self.ROT_GATE_MAP.keys()}."
            )
        return self.ROT_GATE_MAP[c_gate]

    @property
    def params_num(self):
        """
       Property to get the total number of parameters.

       Returns:
           Total number of parameters.
       """
        return self._get_params_num()

    @abstractmethod
    def _get_params_num(self) -> int:
        """
        Abstract method to get the total number of parameters.

        Returns:
            Total number of parameters.
        """
        pass

    @abstractmethod
    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        """
       Abstract method for the variational form function.

       Args:
           params: Variational parameters.
        """
        pass

    @abstractmethod
    def apply(
            self,
            params: np.ndarray
    ) -> None:
        """
        Abstract method to apply the variational form to the quantum circuit.

        Args:
            params: Variational parameters.
        """
        pass


class Ansatz(VariationalForm):
    """
   Quantum variational ansatz based on a user-defined variational function. Given variational
   function needs to include entanglement operations.

   Args:
       n_qubits: Number of qubits in the quantum circuit.
       parameters_num: Number of parameters in the variational function.
       variational_func: User-defined variational function.
       reps: Number of repetitions. Defaults to 1.

   Attributes:
       _expected_args: List of expected arguments for the variational function.
   """

    _expected_args = ['params']

    def __init__(
            self,
            n_qubits: int,
            parameters_num: int,
            variational_func: Callable,
            reps: int = 1
    ):

        self._parameters_num = parameters_num

        super().__init__(
            n_qubits=n_qubits,
            controlled_gate='CNOT',
            reps=reps
        )

        if not validate_function_args(variational_func, self._expected_args):
            raise ValueError(
                f"The variational_form function must "
                f"have the arguments: {self._expected_args}"
            )

        self._variational_function = variational_func

    def _get_params_num(self) -> int:
        """
        Returns the total number of parameters.

        Returns:
            Total number of parameters.
        """
        return self._reps * self._parameters_num

    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        """
        Calls the user-defined variational function.

        Args:
            params: Variational parameters.
        """
        return self._variational_function(params)

    def apply(
            self,
            params: np.ndarray
    ) -> None:
        """
        Applies the variational form to the quantum circuit.

        Args:
            params: Variational parameters.
        """
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
    """
    The two-local circuit is a parameterized circuit consisting of alternating rotation layers
    and entanglement layers. The rotation layers are single qubit gates applied on all qubits.
    The entanglement layer uses two-qubit gates to entangle the qubits.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        rotation_blocks: List of rotation gate types.
        controlled_gate: Controlled gate type. Defaults to 'CX'.
        entanglement: Entanglement pattern. Defaults to 'linear'.
        reps: Number of repetitions. Defaults to 1.
    """
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CNOT',
            entanglement: Union[str, list[list[int]]] = 'linear',
            reps: int = 1
    ):
        super().__init__(
            n_qubits=n_qubits,
            controlled_gate=controlled_gate,
            reps=reps
        )

        self._entanglement = Entangler(
            n_qubits=n_qubits,
            c_gate=controlled_gate,
            entanglement=entanglement
        )

        self._rotation_blocks = [
            self._validate_gate(block_gate)
            for block_gate in rotation_blocks or ['RY']
        ]

    def _get_params_num(self) -> int:
        """
        Returns the total number of parameters.

        Returns:
            Total number of parameters.
        """
        rot_block_n = len(self._rotation_blocks)
        return self._reps * self._n_qubits * rot_block_n

    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        """
        Defines the variational form.

        Args:
            params: Variational parameters.
        """
        for j, rot_ in enumerate(self._rotation_blocks):
            for q in range(self._n_qubits):
                rot_(params[j * self._n_qubits + q], wires=[q])
        self._entanglement.apply()

    def apply(
            self,
            params: np.ndarray
    ) -> None:
        """
        Applies the variational form to the quantum circuit.

        Args:
            params: Variational parameters.
        """
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
    """
    The EfficientSU2 circuit consists of layers of single qubit operations spanned by SU(2) and
    CX entanglements. This is a heuristic pattern that can be used to prepare trial wave functions
    for variational quantum algorithms or classification circuit for machine learning. SU(2) stands
    for special unitary group of degree 2, its elements are 2×2 unitary matrices with determinant 1,
    such as the Pauli rotation gates.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        rotation_blocks: List of rotation gate types.
        entanglement: Entanglement pattern. Defaults to 'linear'.
        reps: Number of repetitions. Defaults to 1.
    """
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
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
            controlled_gate='CNOT',
            entanglement=entanglement,
            reps=reps
        )


class TreeTensor(VariationalForm):
    """
    Quantum variational form using the TreeTensor ansatz. Fits best in models designed for
    binary classification.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        controlled_gate: Controlled gate type. Defaults to 'CX'.
    """
    def __init__(
            self,
            n_qubits: int,
            controlled_gate: str = 'CNOT'
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
        """
        Returns the total number of parameters.

        Returns:
            Total number of parameters.
        """
        return 2 * self._n_qubits - 1

    def _variational_func(
            self,
            params: np.ndarray
    ) -> None:
        """
        Defines the variational form.

        Args:
            params: Variational parameters.
        """
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
        """
        Applies the variational form to the quantum circuit.

        Args:
            params: Variational parameters.
        """
        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )
        self._variational_func(params=params)
