"""
Gradient computation via parameter shift rule and variance analysis.
"""

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit


def compute_gradient_parameter_shift(circuit, params, param_values,
                                      observable_fn, shift=np.pi / 2,
                                      shots=4096, seed=42):
    """
    Compute the gradient of an expectation value using the parameter
    shift rule: df/dθ = [f(θ+s) - f(θ-s)] / (2 sin(s))

    Args:
        circuit: parameterized QuantumCircuit
        params: ParameterVector
        param_values: numpy array of current parameter values
        observable_fn: function(circuit, shots, seed) -> float expectation
        shift: shift amount (default pi/2 for standard gates)
        shots: number of measurement shots
        seed: random seed

    Returns:
        numpy array of partial derivatives
    """
    n_params = len(param_values)
    gradients = np.zeros(n_params)
    scale = 2 * np.sin(shift)

    for i in range(n_params):
        # forward shift
        shifted_plus = param_values.copy()
        shifted_plus[i] += shift
        bound_plus = circuit.assign_parameters(
            dict(zip(params, shifted_plus))
        )
        val_plus = observable_fn(bound_plus, shots, seed)

        # backward shift
        shifted_minus = param_values.copy()
        shifted_minus[i] -= shift
        bound_minus = circuit.assign_parameters(
            dict(zip(params, shifted_minus))
        )
        val_minus = observable_fn(bound_minus, shots, seed)

        gradients[i] = (val_plus - val_minus) / scale

    return gradients


def compute_gradient_variance(ansatz_builder, n_qubits, n_layers,
                               observable_fn, n_samples=50, shots=4096,
                               seed=42):
    """
    Estimate gradient variance by sampling random parameter vectors
    and computing partial derivatives for a single parameter.

    Returns variance of the gradient of the first parameter.
    """
    rng = np.random.RandomState(seed)
    circuit, params = ansatz_builder(n_qubits, n_layers)
    n_params = len(params)

    grad_samples = []
    for trial in range(n_samples):
        param_values = rng.uniform(0, 2 * np.pi, size=n_params)

        # compute gradient of first parameter only (representative)
        shift = np.pi / 2
        shifted_plus = param_values.copy()
        shifted_plus[0] += shift
        shifted_minus = param_values.copy()
        shifted_minus[0] -= shift

        bound_plus = circuit.assign_parameters(dict(zip(params, shifted_plus)))
        bound_minus = circuit.assign_parameters(dict(zip(params, shifted_minus)))

        val_plus = observable_fn(bound_plus, shots, seed + trial)
        val_minus = observable_fn(bound_minus, shots, seed + trial)

        grad = (val_plus - val_minus) / 2.0
        grad_samples.append(grad)

    return np.var(grad_samples), np.mean(grad_samples)


def sweep_gradient_variance(ansatz_builder, qubit_range, n_layers,
                             observable_fn, n_samples=50, shots=4096,
                             seed=42):
    """
    Sweep gradient variance across different qubit counts.

    Args:
        ansatz_builder: function(n_qubits, n_layers) -> (circuit, params)
        qubit_range: list of qubit counts to test
        n_layers: fixed number of layers
        observable_fn: callable for expectation value
        n_samples: random parameter samples per qubit count
        shots: measurement shots
        seed: random seed

    Returns:
        dict with 'n_qubits', 'variances', 'means'
    """
    variances = []
    means = []

    for n_q in qubit_range:
        print(f"  n_qubits={n_q}...", end=" ")
        var, mean = compute_gradient_variance(
            ansatz_builder, n_q, n_layers, observable_fn,
            n_samples=n_samples, shots=shots, seed=seed
        )
        variances.append(var)
        means.append(mean)
        print(f"var={var:.6f}")

    return {
        'n_qubits': list(qubit_range),
        'variances': variances,
        'means': means,
    }


def sweep_depth_variance(ansatz_builder, n_qubits, layer_range,
                          observable_fn, n_samples=50, shots=4096,
                          seed=42):
    """
    Sweep gradient variance across different circuit depths.

    Returns:
        dict with 'n_layers', 'variances', 'means'
    """
    variances = []
    means = []

    for n_l in layer_range:
        print(f"  n_layers={n_l}...", end=" ")
        var, mean = compute_gradient_variance(
            ansatz_builder, n_qubits, n_l, observable_fn,
            n_samples=n_samples, shots=shots, seed=seed
        )
        variances.append(var)
        means.append(mean)
        print(f"var={var:.6f}")

    return {
        'n_layers': list(layer_range),
        'variances': variances,
        'means': means,
    }
