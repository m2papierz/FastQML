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
Module providing functionalities for feature maps for encoding classical data into quantum states.
"""

from abc import abstractmethod
from itertools import combinations

from typing import Union
from typing import Callable

import pennylane as qml
from jax import numpy as jnp

from fast_qml.quantum_circuits.utils import validate_function_args


class FeatureMap:
    """
    Abstract base class for quantum feature maps.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        map_func: Custom mapping function. Defaults to None.
    """

    def __init__(
            self,
            n_qubits: int,
            map_func: Callable = None
    ):
        self._n_qubits = n_qubits

        if map_func is None:
            self._map_func = self._set_map_func
        else:
            if not validate_function_args(map_func, ['features']):
                raise ValueError(
                    f"The variational_form function must "
                    f"have the arguments: {['features']}"
                )

            self._map_func = map_func

    @abstractmethod
    def _set_map_func(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Abstract method to set the custom mapping function.

        Args:
            features: Input features.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def apply(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Applies the feature map to the given classical features.

        Args:
            features: Input features.
        """
        self._map_func(features=features)


class AngleEmbedding(FeatureMap):
    """
    Quantum feature map using the AngleEmbedding scheme.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        rotation: Rotation gate type. Defaults to 'X'.
    """

    def __init__(
            self,
            n_qubits: int,
            rotation: str = 'X'
    ):
        super().__init__(n_qubits=n_qubits)
        self._rotation = rotation

    def _set_map_func(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Sets the mapping function for AngleEmbedding.

        Args:
            features: Input features.
        """
        def map_func():
            qml.AngleEmbedding(
                features=features,
                wires=range(self._n_qubits),
                rotation=self._rotation
            )
        return map_func()


class AmplitudeEmbedding(FeatureMap):
    def __init__(
            self,
            n_qubits: int,
            normalize: bool = True,
            pad_with: Union[float, complex] = 0.0
    ):
        super().__init__(n_qubits=n_qubits)
        self._normalize = normalize
        self._pad_with = pad_with

    def _set_map_func(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Sets the mapping function for AmplitudeEmbedding.

        Args:
            features: Input features.
        """
        def map_func():
            qml.AmplitudeEmbedding(
                features=features,
                wires=range(self._n_qubits),
                normalize=self._normalize,
                pad_with=self._pad_with
            )
        return map_func()


class IQPEmbedding(FeatureMap):
    def __init__(
            self,
            n_qubits: int,
            n_repeats: int = 1
    ):
        super().__init__(n_qubits=n_qubits)
        self._n_repeats = n_repeats

    def _set_map_func(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Sets the mapping function for IQPEmbedding.

        Args:
            features: Input features.
        """
        def map_func():
            qml.IQPEmbedding(
                features=features,
                wires=range(self._n_qubits),
                n_repeats=self._n_repeats
            )
        return map_func()


class ZZFeatureMap(FeatureMap):
    def __init__(
            self,
            n_qubits: int
    ):
        super().__init__(n_qubits=n_qubits)

    def _verify_data_dims(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Verifies the dimensions of the input features.

        Args:
            features: Input features.

        Raises:
            ValueError: If the dimensions are not valid.
        """
        if features.shape[-1] > self._n_qubits:
            raise ValueError(
                f"Features must be of length {self._n_qubits} or "
                f"less, got length {features.shape[-1]}."
            )

    def _access_features(
            self,
            features: jnp.ndarray,
            idx: int
    ) -> jnp.ndarray:
        """
        Accesses specific features from an array based on the given index. This method selects features
        from the given array, handling both batched and unbatched scenarios.

        Args:
            features: The array of features.
            idx: The index of the feature to access.

        Returns:

        """
        if self._batched:
            return features[:, idx]
        else:
            return features[idx]

    def _set_map_func(
            self,
            features: jnp.ndarray
    ) -> None:
        """
        Sets the mapping function for ZZFeatureMap.

        Args:
            features: Input features.
        """
        self._batched = qml.math.ndim(features) > 1
        self._verify_data_dims(features)

        def map_func():
            n_load = min(features.shape[-1], self._n_qubits)
            for i in range(n_load):
                features_ = self._access_features(features, i)

                qml.Hadamard(wires=[i])
                qml.RZ(2.0 * features_, wires=[i])

            for q0, q1 in list(combinations(range(n_load), 2)):
                f_q0 = self._access_features(features, q0)
                f_q1 = self._access_features(features, q1)

                qml.CZ(wires=[q0, q1])
                qml.RZ( 2.0 * (jnp.pi - f_q0) * (jnp.pi - f_q1), wires=[q1] )
                qml.CZ(wires=[q0, q1])
        return map_func()
