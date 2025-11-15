"""
Parameterized quantum circuit ansatz builders for barren plateau analysis.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector


def build_hardware_efficient_ansatz(n_qubits, n_layers, seed=None):
    """
    Hardware-efficient ansatz with Ry-Rz single-qubit layers and
    linear CNOT entangling layers.

    This is the standard ansatz known to exhibit barren plateaus
    at moderate depth.
    """
    n_params = n_qubits * 2 * n_layers
    params = ParameterVector('θ', n_params)
    qc = QuantumCircuit(n_qubits)

    idx = 0
    for layer in range(n_layers):
        # single-qubit rotations
        for qubit in range(n_qubits):
            qc.ry(params[idx], qubit)
            idx += 1
            qc.rz(params[idx], qubit)
            idx += 1

        # linear entangling layer
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)

    return qc, params


def build_structured_ansatz(n_qubits, n_layers, seed=None):
    """
    Structured ansatz with alternating Rx-Rz layers and circular
    entanglement. Preserves some symmetry structure that can help
    avoid barren plateaus.
    """
    n_params = n_qubits * 2 * n_layers
    params = ParameterVector('θ', n_params)
    qc = QuantumCircuit(n_qubits)

    idx = 0
    for layer in range(n_layers):
        # alternating rotation axes per layer
        for qubit in range(n_qubits):
            qc.rx(params[idx], qubit)
            idx += 1
            qc.rz(params[idx], qubit)
            idx += 1

        # circular entanglement
        for qubit in range(n_qubits):
            qc.cx(qubit, (qubit + 1) % n_qubits)

    return qc, params


def build_identity_block_ansatz(n_qubits, n_layers, seed=None):
    """
    Identity-block initialization ansatz. Each layer is constructed so
    that at initialization (params=0), consecutive pairs of layers
    cancel out, making the circuit close to identity.

    This strategy mitigates barren plateaus by starting near a known
    good point.
    """
    n_params = n_qubits * 2 * n_layers
    params = ParameterVector('θ', n_params)
    qc = QuantumCircuit(n_qubits)

    idx = 0
    for layer in range(n_layers):
        # forward rotations
        for qubit in range(n_qubits):
            qc.ry(params[idx], qubit)
            idx += 1

        # entangling layer
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)

        # reverse rotations (negative params create identity at init)
        for qubit in range(n_qubits):
            qc.ry(-params[idx], qubit)
            idx += 1

        # reverse entangling
        for qubit in range(n_qubits - 2, -1, -1):
            qc.cx(qubit, qubit + 1)

    return qc, params


def get_ansatz_builders():
    """Return dict of ansatz builder functions."""
    return {
        'Hardware-Efficient': build_hardware_efficient_ansatz,
        'Structured': build_structured_ansatz,
        'Identity-Block': build_identity_block_ansatz,
    }
