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
from itertools import combinations
from typing import Union, Callable

import numpy as np
import pennylane as qml


class FeatureMap:
    def __init__(
            self,
            n_qubits: int,
            map_func: Callable = None
    ):
        self._n_qubits = n_qubits

        if map_func is None:
            self._map_func = self._set_map_func
        else:
            self._map_func = map_func

    @abstractmethod
    def _set_map_func(
            self,
            features: np.ndarray
    ):
        pass

    def apply(
            self,
            features: np.ndarray
    ) -> None:
        self._map_func(features=features)


class AngleEmbedding(FeatureMap):
    def __init__(
            self,
            n_qubits: int,
            rotation: str = 'X'
    ):
        super().__init__(n_qubits=n_qubits)
        self._rotation = rotation

    def _set_map_func(
            self,
            features: np.ndarray
    ):
        def map_func():
            qml.AngleEmbedding(
                features=features,
                wires=range(self._n_qubits),
                rotation=self._rotation
            )
        return map_func


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
            features: np.ndarray
    ):
        def map_func():
            qml.AmplitudeEmbedding(
                features=features,
                wires=range(self._n_qubits),
                normalize=self._normalize,
                pad_with=self._pad_with
            )
        return map_func


class IQPEmbedding(FeatureMap):
    def __init__(
            self,
            n_qubits: int
    ):
        super().__init__(n_qubits=n_qubits)

    def _set_map_func(
            self,
            features: np.ndarray
    ):
        def map_func():
            qml.IQPEmbedding(
                features=features,
                wires=range(self._n_qubits)
            )
        return map_func


class ZZFeatureMap(FeatureMap):
    def __init__(
            self,
            n_qubits: int
    ):
        super().__init__(n_qubits=n_qubits)

    def _verify_data_dims(
            self,
            features: np.ndarray
    ) -> None:
        if features.shape[-1] > self._n_qubits:
            raise ValueError(
                f"Features must be of length {self._n_qubits} or "
                f"less, got length {features.shape[-1]}."
            )

    def _set_map_func(
            self,
            features: np.ndarray
    ):
        def map_func():
            self._verify_data_dims(features)

            n_load = min(features.shape[-1], self._n_qubits)
            for i in range(n_load):
                qml.Hadamard(wires=[i])
                qml.RZ(2.0 * features[:, i], wires=[i])

            for q0, q1 in list(combinations(range(n_load), 2)):
                qml.CZ(wires=[q0, q1])
                qml.RZ(
                    2.0 * (np.pi - features[:, q0]) * (np.pi - features[:, q1]),
                    wires=[q1]
                )
                qml.CZ(wires=[q0, q1])
        return map_func
