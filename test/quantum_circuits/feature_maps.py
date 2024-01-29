import numpy as np
import pennylane as qml

from fast_qml.quantum_circuits.feature_maps import (
    AngleEmbedding, AmplitudeEmbedding, IQPEmbedding, ZZFeatureMap
)


def test_angle_embedding():
    """Test the AngleEmbedding feature map."""

    n_qubits = 3
    features = np.array([0.1, 0.2, 0.3])
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def _test_circuit():
        AngleEmbedding(n_qubits=n_qubits).apply(features)
        return qml.state()

    @qml.qnode(dev)
    def _reference_circuit():
        qml.AngleEmbedding(features, wires=range(n_qubits))
        return qml.state()

    assert np.allclose(_test_circuit(), _reference_circuit())


def test_amplitude_embedding():
    """Test the AmplitudeEmbedding feature map."""

    n_qubits = 2
    features = np.array([0.1, 0.2, 0.7, 0.0])
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def _test_circuit():
        AmplitudeEmbedding(n_qubits=n_qubits).apply(features)
        return qml.state()

    @qml.qnode(dev)
    def _reference_circuit():
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
        return qml.state()

    assert np.allclose(_test_circuit(), _reference_circuit())


def test_iqp_embedding():
    """Test the IQPEmbedding feature map."""

    n_qubits = 3
    features = np.array([0.1, 0.2, 0.3])
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def _test_circuit():
        IQPEmbedding(n_qubits=n_qubits).apply(features)
        return qml.state()

    @qml.qnode(dev)
    def _reference_circuit():
        qml.IQPEmbedding(features, wires=range(n_qubits))
        return qml.state()

    assert np.allclose(_test_circuit(), _reference_circuit())


def test_zz_feature_map():
    """Test the ZZFeatureMap feature map."""

    def reference_zz_feature_map_circ(features):
        for i in range(n_qubits):
            qml.Hadamard(wires=[i])
            qml.RZ(2.0 * features[:, i], wires=[i])

        qml.CZ(wires=[0, 1])
        qml.RZ(2.0 * (np.pi - features[:, 0]) * (np.pi - features[:, 1]), wires=[1])
        qml.CZ(wires=[0, 1])

        qml.CZ(wires=[0, 2])
        qml.RZ(2.0 * (np.pi - features[:, 0]) * (np.pi - features[:, 2]), wires=[2])
        qml.CZ(wires=[0, 2])

        qml.CZ(wires=[1, 2])
        qml.RZ(2.0 * (np.pi - features[:, 1]) * (np.pi - features[:, 2]), wires=[2])
        qml.CZ(wires=[1, 2])

    n_qubits = 3
    features = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def _test_circuit():
        ZZFeatureMap(n_qubits=n_qubits).apply(features)
        return qml.state()

    @qml.qnode(dev)
    def _reference_circuit():
        reference_zz_feature_map_circ(features)
        return qml.state()

    assert np.allclose(_test_circuit(), _reference_circuit())
