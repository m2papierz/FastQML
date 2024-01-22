import jax
import jaxopt
import pennylane as qml

from typing import Callable
from abc import abstractmethod

from fast_qml import QUBIT_DEV
from fast_qml import numpy as np
from fast_qml.quantum_circuits.feature_maps import FeatureMap
from fast_qml.quantum_circuits.variational_forms import VariationalForm


class VariationalEstimator:
    def __init__(
            self,
            n_qubits: int,
            feature_map: FeatureMap,
            ansatz: VariationalForm,
            measurement_operator: qml.Operation,
            loss_fn: Callable,
            n_layers: int
    ):
        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._ansatz = ansatz
        self._measurement_op = measurement_operator
        self._loss_fn = loss_fn

        self._ansatz._reps = n_layers
        self._device = qml.device(QUBIT_DEV, wires=n_qubits)

        self._params = self._init_weights()

    @abstractmethod
    def _init_weights(self):
        pass

    def _circuit(
            self,
            weights: np.ndarray,
            x_data: np.ndarray
    ):
        @qml.qnode(device=self._device)
        def _quantum_circuit():
            self._feature_map.circuit(features=x_data)
            self._ansatz.circuit(params=weights)
            return qml.expval(op=self._measurement_op)
        return _quantum_circuit()

    @jax.jit
    def loss_fn(self, weights, x_data, y_data):
        predictions = self._circuit(
            weights=weights, x_data=x_data)
        return np.sum(
            (y_data - predictions) ** 2 / len(x_data)
        )

    @jax.jit
    def _optimization_loop_cpu(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            epochs: int,
            verbose: bool = False
    ):
        def loss_and_grad():
            loss_val, grad_val = jax.value_and_grad(
                self._loss_fn)(self._params, x_data, y_data)

            if verbose:
                jax.debug.print(f"Step: {i} - Loss: {loss_val}")

            return loss_val, grad_val

        opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.1, value_and_grad=True)
        opt_state = opt.init_state(self._params)

        for i in range(epochs):
            self._params, opt_state = opt.update(
                self._params, opt_state, x_data, y_data, True, i
            )
