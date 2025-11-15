"""
Cost function observables for barren plateau experiments.

Global cost functions measure all qubits and are known to cause
barren plateaus. Local cost functions measure only a subset of
qubits and can have more favorable gradient scaling.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def _measure_expectation(circuit, shots, seed, measured_qubits=None):
    """
    Measure expectation value of Z operator on specified qubits.
    Uses Pauli-Z: maps |0> -> +1 and |1> -> -1.
    """
    n_qubits = circuit.num_qubits
    if measured_qubits is None:
        measured_qubits = list(range(n_qubits))

    from qiskit.circuit import ClassicalRegister
    meas_circuit = circuit.copy()
    cr = ClassicalRegister(len(measured_qubits), 'meas')
    meas_circuit.add_register(cr)
    for i, q in enumerate(measured_qubits):
        meas_circuit.measure(q, cr[i])

    backend = AerSimulator(seed_simulator=seed)
    job = backend.run(meas_circuit, shots=shots)
    counts = job.result().get_counts()

    # compute <Z^n> expectation
    expectation = 0.0
    for bitstring, count in counts.items():
        # each bit contributes +1 for |0> or -1 for |1>
        parity = sum(int(b) for b in bitstring) % 2
        sign = (-1) ** parity
        expectation += sign * count

    return expectation / shots


def global_cost_observable(circuit, shots=4096, seed=42):
    """
    Global cost function: measures Z on ALL qubits.

    C_global = <0|U†(θ) (Z⊗Z⊗...⊗Z) U(θ)|0>

    Known to exhibit barren plateaus with exponentially vanishing
    gradients as qubit count increases.
    """
    n_qubits = circuit.num_qubits
    measured_qubits = list(range(n_qubits))
    return _measure_expectation(circuit, shots, seed, measured_qubits)


def local_cost_observable(circuit, shots=4096, seed=42):
    """
    Local cost function: measures Z on only the FIRST qubit.

    C_local = <0|U†(θ) (Z⊗I⊗...⊗I) U(θ)|0>

    Has polynomially vanishing gradients for shallow circuits,
    making it more trainable than the global cost.
    """
    return _measure_expectation(circuit, shots, seed, measured_qubits=[0])


def two_local_cost_observable(circuit, shots=4096, seed=42):
    """
    Two-local cost function: measures Z on first two qubits.

    C_2local = <0|U†(θ) (Z⊗Z⊗I⊗...⊗I) U(θ)|0>

    Intermediate between global and local.
    """
    n_qubits = circuit.num_qubits
    if n_qubits < 2:
        return local_cost_observable(circuit, shots, seed)
    return _measure_expectation(circuit, shots, seed, measured_qubits=[0, 1])
