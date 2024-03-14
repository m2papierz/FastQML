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
from typing import Any
from typing import Union
from typing import Callable
from typing import List

import pennylane as qml
from jax import numpy as jnp
from pennylane.ops.op_math.controlled import ControlledOp

from fast_qml.quantum_circuits.utils import validate_function_args
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
        """ Validates the rotation gate type. """
        if c_gate not in self.ROT_GATE_MAP:
            raise ValueError(
                f"Invalid rotation gate type. "
                f"Supported types are {self.ROT_GATE_MAP.keys()}."
            )
        return self.ROT_GATE_MAP[c_gate]

    @property
    def params_num(self):
        """
        Returns the total number of parameters.
        """
        return self._get_params_num()

    @abstractmethod
    def _get_params_num(self) -> int:
        """
        Abstract method for calculating the total number of parameters required for the ansatz.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _variational_func(
            self,
            params: jnp.ndarray
    ) -> None:
        """
        Abstract method for the variational form function.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def apply(
            self,
            params: jnp.ndarray
    ) -> None:
        """
        Abstract method to apply the variational form to the quantum circuit.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Ansatz(VariationalForm):
    """
    This class represents a quantum variational ansatz based on a user-defined variational function.
    It constructs a quantum circuit with a specified number of qubits, controlled gates, and entanglement
    pattern, repeated a given number of times. The variational function defines the rotation layers
    within the quantum circuit.

   Args:
       n_qubits: Number of qubits in the quantum circuit.
       parameters_num: Number of parameters in the variational function.
       variational_func: User-defined variational function. It must accept a single argument 'params'.
       entanglement: Entanglement pattern. Defaults to 'linear'.
       controlled_gate: Controlled gate type. Defaults to 'CNOT'.
       skip_last_rotations: If True, skips the rotation layer in the last repetition. Defaults to False.
       reps: Number of repetitions of the variational function and entanglement pattern. Defaults to 1.

    **Example**

    # Import necessary libraries
    >>> import numpy as np
    >>> from fast_qml.quantum_circuits.ansatz import Ansatz

    # Define a user-specific variational function
    >>> def my_variational_function(params):
    ...     qml.RX(params[0], wires=[0])
    ...     qml.RY(params[1], wires=[1])
    ...     qml.RX(params[2], wires=[2])
    ...     qml.RY(params[3], wires=[3])
    ...     qml.RY(params[4], wires=[0])
    ...     qml.RX(params[5], wires=[2])

    # Initialize parameters for the Ansatz
    >>> n_qubits = 4
    >>> parameters_num = 6
    >>> reps = 2

    # Create an instance of the Ansatz class
    >>> ansatz = Ansatz(
    ...     n_qubits=n_qubits,
    ...     parameters_num=parameters_num,
    ...     variational_func=my_variational_function,
    ...     entanglement='reverse_circular',
    ...     controlled_gate='CNOT',
    ...     skip_last_rotations=False,
    ...     reps=reps
    ... )

    # Initialize randomly parameters and draw circuit
    >>> params = np.random.randn(ansatz.params_num)
    >>> print(qml.draw(ansatz.apply)(params))
    0: ──RX(-0.18)──RY(-0.95)───────╭X─╭●──||──RX(0.89)───RY(0.64)───────╭X─╭●──||──RX(-0.68)──RY(-0.53)─┤
    1: ──RY(-0.79)───────────────╭X─╰●─│───||──RY(1.38)───────────────╭X─╰●─│───||──RY(0.40)─────────────┤
    2: ──RX(0.31)───RX(0.19)──╭X─╰●────│───||──RX(-0.11)──RX(0.47)─╭X─╰●────│───||──RX(0.02)───RX(0.37)──┤
    3: ──RY(-0.28)────────────╰●───────╰X──||──RY(-1.47)───────────╰●───────╰X──||──RY(0.12)─────────────┤
   """

    def __init__(
            self,
            n_qubits: int,
            parameters_num: int,
            variational_func: Callable,
            entanglement: Union[str, list[list[int]]] = 'linear',
            controlled_gate: str = 'CNOT',
            skip_last_rotations: bool = False,
            reps: int = 1
    ):

        self._parameters_num = parameters_num

        super().__init__(
            n_qubits=n_qubits,
            controlled_gate='CNOT',
            reps=reps
        )

        if not validate_function_args(variational_func, ['params']):
            raise ValueError(
                "The variational_func must accept 'params' as an argument."
            )

        self._variational_function = variational_func
        self._skip_last_rotations = skip_last_rotations

        self._entanglement = Entangler(
            n_qubits=n_qubits,
            c_gate=controlled_gate,
            entanglement=entanglement
        )

    def _get_params_num(self) -> int:
        """
        Calculates the total number of parameters required for the ansatz.
        """
        add_rep = 0 if self._skip_last_rotations else 1
        return (self._reps + add_rep) * self._parameters_num

    def _apply_rotation_layer(
            self, params: jnp.ndarray
    ) -> None:
        """
        Applies the user-defined rotation layer to the quantum circuit.

        Args:
            params: Array of parameters for the rotation layer.
        """
        self._variational_function(params)

    def _variational_func(
            self,
            params: jnp.ndarray
    ) -> None:
        """
        Defines and applies the variational form of the quantum circuit.

        Args:
            params: Array of parameters for the variational function.
        """
        self._apply_rotation_layer(params=params)
        self._entanglement.apply()
        qml.Barrier()

    def apply(
            self,
            params: Union[jnp.ndarray, None] = None
    ) -> None:
        """
        Applies the variational ansatz to the quantum circuit.

        Args:
            params: Array of parameters for the entire variational form.
        """
        if params is None:
            params = jnp.zeros(self._get_params_num())

        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )

        if self._skip_last_rotations:
            block_params_n = int(self._get_params_num() / self._reps)
        else:
            block_params_n = int(self._get_params_num() / (self._reps + 1))

        for r in range(self._reps):
            params_subset = params[r * block_params_n: (r + 1) * block_params_n]
            self._variational_func(params=params_subset)

        if not self._skip_last_rotations:
            last_params_subset = params[
                            self._reps * block_params_n:
                            (self._reps + 1) * block_params_n
                        ]
            self._apply_rotation_layer(params=last_params_subset)

class TwoLocal(VariationalForm):
    """
    The two-local circuit is a parameterized circuit consisting of alternating rotation layers
    and entanglement layers. The rotation layers are single qubit gates applied on all qubits.
    The entanglement layer uses two-qubit gates to entangle the qubits.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        rotation_blocks: List of rotation gate types.
        controlled_gate: Controlled gate type. Defaults to 'CNOT'.
        entanglement: Entanglement pattern. Defaults to 'linear'.
        skip_last_rotations (bool): Whether to skip rotations in the last repetition, default is False.
        reps: Number of repetitions. Defaults to 1.

    **Example**

    # Import necessary libraries
    >>> import numpy as np
    >>> from fast_qml.quantum_circuits.ansatz import TwoLocal

    # Create an instance of the TwoLocal class
    >>> two_local = TwoLocal(
    ...     n_qubits=4,
    ...     controlled_gate='CZ',
    ...     entanglement='circular',
    ...     skip_last_rotations=False,
    ...     reps=1
    ... )

    # Initialize randomly parameters and draw circuit
    >>> params = np.random.randn(two_local.params_num)
    >>> print(qml.draw(two_local.apply)(params))
    0: ──RY(-1.39)─╭●───────╭Z──||──RY(-0.78)─┤
    1: ──RY(-0.43)─╰Z─╭●────│───||──RY(0.70)──┤
    2: ──RY(-1.07)────╰Z─╭●─│───||──RY(0.25)──┤
    3: ──RY(1.27)────────╰Z─╰●──||──RY(-0.70)─┤
    """
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            controlled_gate: str = 'CNOT',
            entanglement: Union[str, list[list[int]]] = 'linear',
            skip_last_rotations: bool = False,
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

        self._skip_last_rotations = skip_last_rotations

    def _get_params_num(self) -> int:
        """
        Calculates the total number of parameters required for the ansatz.
        """
        add_rep = 0 if self._skip_last_rotations else 1
        rot_block_n = len(self._rotation_blocks)
        return (self._reps + add_rep) * self._n_qubits * rot_block_n

    def _apply_rotation_layer(
            self, params: jnp.ndarray
    ) -> None:
        """
        Applies rotation layer to the quantum circuit.

        Args:
            params: Array of parameters for the rotation layer.
        """
        for j, rot_ in enumerate(self._rotation_blocks):
            for q in range(self._n_qubits):
                rot_(params[j * self._n_qubits + q], wires=[q])

    def _variational_func(
            self,
            params: jnp.ndarray
    ) -> None:
        """
        Defines and applies the variational form of the quantum circuit.

        Args:
            params: Array of parameters for the variational function.
        """
        self._apply_rotation_layer(params=params)
        self._entanglement.apply()
        qml.Barrier()

    def apply(
            self,
            params: Union[jnp.ndarray] = None
    ) -> None:
        """
        Applies the variational ansatz to the quantum circuit.

        Args:
            params: Array of parameters for the entire variational form.
        """
        if params is None:
            params = jnp.zeros(self._get_params_num())

        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )

        if self._skip_last_rotations:
            block_params_n = int(self._get_params_num() / self._reps)
        else:
            block_params_n = int(self._get_params_num() / (self._reps + 1))

        for r in range(self._reps):
            params_subset = params[r * block_params_n: (r + 1) * block_params_n]
            self._variational_func(params=params_subset)

        if not self._skip_last_rotations:
            last_params_subset = params[
                            self._reps * block_params_n:
                            (self._reps + 1) * block_params_n
                        ]
            self._apply_rotation_layer(params=last_params_subset)


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

    **Example**

    # Import necessary libraries
    >>> import numpy as np
    >>> from fast_qml.quantum_circuits.ansatz import EfficientSU2

    # Create an instance of the EfficientSU2 class
    >>> su2 = EfficientSU2(
    ...     n_qubits=3,
    ...     rotation_blocks=['RX', 'RY'],
    ...     entanglement='circular',
    ...     skip_last_rotations=True,
    ...     reps=2
    ... )

    # Initialize randomly parameters and draw circuit
    >>> params = np.random.randn(su2.params_num)
    >>> print(qml.draw(su2.apply)(params))
    0: ──RX(-1.17)──RY(0.60)──╭●────╭X──||──RX(-1.32)──RY(1.19)──╭●────╭X──||─┤
    1: ──RX(-0.77)──RY(-0.65)─╰X─╭●─│───||──RX(0.25)───RY(-1.38)─╰X─╭●─│───||─┤
    2: ──RX(-0.98)──RY(-1.45)────╰X─╰●──||──RX(-0.14)──RY(-0.55)────╰X─╰●──||─┤
    """
    def __init__(
            self,
            n_qubits: int,
            rotation_blocks: list[str] = None,
            entanglement: str = 'linear',
            skip_last_rotations: bool = False,
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
            skip_last_rotations=skip_last_rotations,
            reps=reps
        )


class TreeTensor(VariationalForm):
    """
    Quantum variational form using the TreeTensor ansatz, suitable
    for binary classification models.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        controlled_gate: Controlled gate type. Defaults to 'CNOT'.

    **Example**

    # Import necessary libraries
    >>> import numpy as np
    >>> from fast_qml.quantum_circuits.ansatz import TreeTensor

    # Create an instance of the TreeTensor class
    >>> tree_tensor = TreeTensor(
    ...     n_qubits=4,
    ...     controlled_gate='CNOT',
    ...     reps=2
    ... )

    # Initialize randomly parameters and draw circuit
    >>> params = np.random.randn(tree_tensor.params_num)
    >>> print(qml.draw(tree_tensor.apply)(params))
    0: ──RY(-0.11)─╭●──RY(0.77)──╭●──RY(-1.06)──||──RY(0.77)──╭●──RY(-0.29)─╭●──RY(0.10)──||─┤
    1: ──RY(0.38)──╰X────────────│──────────────||──RY(-2.16)─╰X────────────│─────────────||─┤
    2: ──RY(-0.17)─╭●──RY(-0.66)─╰X─────────────||──RY(-0.24)─╭●──RY(1.02)──╰X────────────||─┤
    3: ──RY(0.80)──╰X───────────────────────────||──RY(-1.17)─╰X──────────────────────────||─┤
    """
    def __init__(
            self,
            n_qubits: int,
            controlled_gate: str = 'CNOT',
            reps: int = 1
    ):
        # Check if n_qubits is a power of two
        if not (n_qubits & (n_qubits - 1)) == 0:
            raise ValueError(
                "TreeTensor ansatz requires the number of qubits "
                "to be a power of two."
            )

        self._layers = int(jnp.log2(n_qubits))

        super().__init__(
            n_qubits=n_qubits,
            controlled_gate=controlled_gate,
            reps=reps
        )

    def _get_params_num(self) -> int:
        """
        Calculates the total number of parameters required for the ansatz.
        """
        return (2 * self._n_qubits - 1) * self._layers

    def _variational_func(
            self,
            params: jnp.ndarray
    ) -> None:
        """
        Defines and applies the variational form of the quantum circuit.

        Args:
            params: Array of parameters for the variational function.
        """
        for i in range(self._n_qubits):
            qml.RY(params[i], wires=[i])

        n_qubits = self._n_qubits
        for r in range(1, self._layers + 1):
            for s in range(0, 2 ** (self._layers - r)):
                target_wire = (s * 2 ** r)
                control_wire = target_wire + (2 ** (r - 1))

                qml.CNOT(wires=[target_wire, control_wire])
                qml.RY(params[n_qubits + s], wires=[target_wire])

            n_qubits += 2 ** (self._layers - r)
        qml.Barrier()

    def apply(
            self,
            params: Union[jnp.ndarray] = None
    ) -> None:
        """
        Applies the variational ansatz to the quantum circuit.

        Args:
            params: Array of parameters for the entire variational form.
        """
        if params is None:
            params = jnp.zeros(self._get_params_num())

        if len(params) != self.params_num:
            ValueError(
                f"Invalid parameters shape. "
                f"Expected {self.params_num}, got {len(params)}."
            )

        block_params_n = int(self._get_params_num() / self._reps)
        for r in range(self._reps):
            params_subset = params[r * block_params_n: (r + 1) * block_params_n]
            self._variational_func(params=params_subset)


class StronglyEntanglingLayers(VariationalForm):
    """
    QLayers consisting of single qubit rotations and entanglers, inspired by the circuit-centric
    classifier design: https://arxiv.org/abs/1804.00633.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        controlled_gate: Controlled gate type. Defaults to 'CNOT'.
        n_layers: Number of strongly entangling layers.
        ranges: Sequence determining the range hyperparameter for each subsequent layer.

    **Example**

    # Import necessary libraries
    >>> import numpy as np
    >>> from fast_qml.quantum_circuits.ansatz import StronglyEntanglingLayers

    # Create an instance of the StronglyEntanglingLayers class
    >>> strong_ent = TreeTensor(
    ...     n_qubits=4,
    ...     controlled_gate='CNOT',
    ...     reps=2
    ... )

    # Create QNode
    >>> @qml.qnode(qml.device("default.qubit"), wires=4)
    >>> def circ(params)
    ...     strong_ent.apply(params)
    ...     return qml.expval(qml.PauliZ(0))

    # Initialize randomly parameters and draw circuit
    >>> params = np.random.randn(strong_ent.params_num)
    >>> print(qml.draw(circ, expansion_strategy='device')(params))
    0: ──Rot(-0.39,0.60,-0.15)─╭●───────╭X──Rot(-1.19,-1.06,-0.35)─╭●────╭X────┤  <Z>
    1: ──Rot(0.95,0.58,-1.11)──╰X─╭●────│───Rot(0.87,-0.78,-0.92)──│──╭●─│──╭X─┤
    2: ──Rot(0.58,-0.70,1.75)─────╰X─╭●─│───Rot(0.72,-0.37,-0.46)──╰X─│──╰●─│──┤
    3: ──Rot(-1.36,-1.99,0.76)───────╰X─╰●──Rot(0.15,0.75,-0.12)──────╰X────╰●─┤
    """
    def __init__(
            self,
            n_qubits: int,
            controlled_gate: str = 'CNOT',
            n_layers: int = 1,
            ranges: List[int] = None
    ):
        super().__init__(
            n_qubits=n_qubits,
            controlled_gate=controlled_gate,
            reps=1
        )

        self._n_layers = n_layers
        self._ranges = ranges
        self._controlled_gate = self._get_controlled_gate(
            c_gate=controlled_gate
        )

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

    def _get_params_num(self) -> int:
        """
        Calculates the total number of parameters required for the ansatz.
        """
        return qml.StronglyEntanglingLayers.shape(
            n_layers=self._n_layers, n_wires=self._n_qubits
        )

    def _variational_func(
            self,
            params: jnp.ndarray
    ) -> None:
        """
        Defines and applies the variational form of the quantum circuit.

        Args:
            params: Array of parameters for the variational function.
        """
        qml.StronglyEntanglingLayers(
            weights=params,
            wires=range(self._n_qubits),
            ranges=self._ranges,
            imprimitive=self._controlled_gate
        )

    def apply(
            self,
            params: Union[jnp.ndarray] = None
    ) -> None:
        """
        Applies the variational ansatz to the quantum circuit.

        Args:
            params: Array of parameters for the entire variational form.
        """
        if params is None:
            params = jnp.zeros(self._get_params_num())

        self._variational_func(params=params)
