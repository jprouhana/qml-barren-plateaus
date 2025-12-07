"""
Training strategies to mitigate barren plateaus:
layer-wise training and parameter initialization schemes.
"""

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA


def random_init(n_params, seed=42):
    """Standard uniform random initialization in [0, 2π)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 2 * np.pi, size=n_params)


def correlated_init(n_params, n_qubits, seed=42):
    """
    Correlated initialization: same rotation angles within each layer.
    Breaks the random symmetry that causes barren plateaus.
    """
    rng = np.random.RandomState(seed)
    params_per_layer = n_qubits * 2  # ry + rz per qubit
    n_layers = n_params // params_per_layer

    init_params = np.zeros(n_params)
    idx = 0
    for layer in range(n_layers):
        # one angle shared across all qubits in the layer
        ry_angle = rng.uniform(0, 2 * np.pi)
        rz_angle = rng.uniform(0, 2 * np.pi)
        for qubit in range(n_qubits):
            init_params[idx] = ry_angle + rng.normal(0, 0.1)
            idx += 1
            init_params[idx] = rz_angle + rng.normal(0, 0.1)
            idx += 1

    return init_params


def identity_init(n_params, seed=42):
    """
    Near-identity initialization: parameters close to zero so the
    circuit is close to the identity operation.
    """
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.01, size=n_params)


def layerwise_train(ansatz_builder, n_qubits, n_layers, cost_fn,
                     maxiter_per_layer=30, shots=4096, seed=42):
    """
    Layer-wise training: train one layer at a time, freezing others.

    This strategy avoids the full barren plateau by optimizing in
    a smaller parameter subspace at each step.

    Args:
        ansatz_builder: function(n_qubits, n_layers) -> (circuit, params)
        n_qubits: number of qubits
        n_layers: total number of layers
        cost_fn: function(circuit, shots, seed) -> float
        maxiter_per_layer: COBYLA iterations per layer
        shots: measurement shots
        seed: random seed

    Returns:
        dict with 'cost_history', 'final_params', 'layer_costs'
    """
    rng = np.random.RandomState(seed)
    full_circuit, full_params = ansatz_builder(n_qubits, n_layers)
    n_params = len(full_params)
    params_per_layer = n_params // n_layers

    # start near identity
    current_params = rng.normal(0, 0.01, size=n_params)
    cost_history = []
    layer_costs = []

    for layer_idx in range(n_layers):
        print(f"    Training layer {layer_idx + 1}/{n_layers}...")

        # define which parameters are trainable in this layer
        start_idx = layer_idx * params_per_layer
        end_idx = start_idx + params_per_layer

        def objective(trainable_params):
            trial_params = current_params.copy()
            trial_params[start_idx:end_idx] = trainable_params
            bound_circuit = full_circuit.assign_parameters(
                dict(zip(full_params, trial_params))
            )
            cost = cost_fn(bound_circuit, shots, seed)
            cost_history.append(cost)
            return -cost  # minimize negative expectation

        x0 = current_params[start_idx:end_idx]
        optimizer = COBYLA(maxiter=maxiter_per_layer)
        result = optimizer.minimize(objective, x0=x0)

        current_params[start_idx:end_idx] = result.x

        # evaluate cost after training this layer
        bound = full_circuit.assign_parameters(
            dict(zip(full_params, current_params))
        )
        layer_cost = cost_fn(bound, shots, seed)
        layer_costs.append(layer_cost)

    return {
        'cost_history': cost_history,
        'final_params': current_params,
        'layer_costs': layer_costs,
    }
