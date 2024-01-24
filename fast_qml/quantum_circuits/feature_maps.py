# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

import pennylane as qml

from abc import abstractmethod
from itertools import combinations
from typing import Union

from jax import numpy as jnp


class FeatureMap:
    def __init__(self, n_qubits: int):
        self._n_qubits = n_qubits

    @abstractmethod
    def apply(
            self,
            features: jnp.ndarray
    ) -> None:
        pass


class AngleEmbedding(FeatureMap):
    def __init__(self, n_qubits, rotation: str = 'X'):
        super().__init__(n_qubits=n_qubits)
        self._rotation = rotation

    def apply(
            self,
            features: jnp.ndarray
    ) -> None:
        qml.AngleEmbedding(
            features=features,
            wires=range(self._n_qubits),
            rotation=self._rotation
        )


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

    def apply(
            self,
            features: jnp.ndarray
    ) -> None:
        qml.AmplitudeEmbedding(
            features=[x for x in features],
            wires=range(self._n_qubits),
            normalize=self._normalize,
            pad_with=self._pad_with
        )


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
        features_len = features.shape[-1]
        if not features_len <= self._n_qubits:
            raise ValueError(
                f"Features must be of length {self._n_qubits}. "
                f"Got length {features_len}."
            )

    def apply(
            self,
            features: jnp.ndarray
    ) -> None:
        self._verify_data_dims(features)

        n_load = min(features.shape[-1], self._n_qubits)
        for i in range(n_load):
            qml.Hadamard(wires=[i])
            qml.RZ(2.0 * features[i], wires=[i])

        for q0, q1 in list(combinations(range(n_load), 2)):
            qml.CZ(wires=[q0, q1])
            qml.RZ(
                2.0 * (jnp.pi - features[q0]) * (jnp.pi - features[q1]),
                wires=[q1]
            )
            qml.CZ(wires=[q0, q1])
