import numpy as np
import pennylane as qml

from abc import abstractmethod
from itertools import combinations
from typing import Union


class FeatureMap:
    def __init__(
            self,
            n_qubits: int,
            features_tensor: np.ndarray
    ):
        self._n_qubits = n_qubits
        self._features_tensor = features_tensor

    @abstractmethod
    def circuit(self) -> None:
        pass

    def __call__(self) -> None:
        self.circuit()


class AngleEmbedding(FeatureMap):
    def __init__(self, n_qubits, features_tensor, rotation: str = 'X'):
        super().__init__(
            n_qubits=n_qubits,
            features_tensor=features_tensor
        )
        self._rotation = rotation

    def circuit(self):
        qml.AngleEmbedding(
            features=self._features_tensor,
            wires=range(self._n_qubits),
            rotation=self._rotation
        )


class AmplitudeEmbedding(FeatureMap):
    def __init__(
            self,
            n_qubits: int,
            features_tensor: np.ndarray,
            normalize: bool = True,
            pad_with: Union[float, complex] = 0.0
    ):
        super().__init__(
            n_qubits=n_qubits,
            features_tensor=features_tensor
        )
        self._normalize = normalize
        self._pad_with = pad_with

    def circuit(self):
        qml.AmplitudeEmbedding(
            features=[x for x in self._features_tensor],
            wires=range(self._n_qubits),
            normalize=self._normalize,
            pad_with=self._pad_with
        )
