"""
Visualization functions for barren plateau analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_gradient_variance_vs_qubits(results_dict, save_dir='results'):
    """
    Log-scale plot of gradient variance vs number of qubits for
    different ansatz types.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Hardware-Efficient': '#FF6B6B', 'Structured': '#4ECDC4',
              'Identity-Block': '#45B7D1'}

    for name, data in results_dict.items():
        ax.semilogy(data['n_qubits'], data['variances'], 'o-',
                     color=colors.get(name, 'gray'), linewidth=2,
                     markersize=8, label=name)

    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Gradient Variance (log scale)', fontsize=12)
    ax.set_title('Barren Plateau: Gradient Variance vs Circuit Width')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'gradient_variance_vs_qubits.png', dpi=150)
    plt.close()


def plot_gradient_variance_vs_depth(results_dict, save_dir='results'):
    """
    Log-scale plot of gradient variance vs circuit depth.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Hardware-Efficient': '#FF6B6B', 'Structured': '#4ECDC4',
              'Identity-Block': '#45B7D1'}

    for name, data in results_dict.items():
        ax.semilogy(data['n_layers'], data['variances'], 's-',
                     color=colors.get(name, 'gray'), linewidth=2,
                     markersize=8, label=name)

    ax.set_xlabel('Number of Layers (Depth)', fontsize=12)
    ax.set_ylabel('Gradient Variance (log scale)', fontsize=12)
    ax.set_title('Barren Plateau: Gradient Variance vs Circuit Depth')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'gradient_variance_vs_depth.png', dpi=150)
    plt.close()


def plot_variance_heatmap(variance_matrix, qubit_range, layer_range,
                           title='Gradient Variance Heatmap',
                           save_dir='results'):
    """
    Heatmap of gradient variance as a function of both qubits and depth.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # log10 for better visualization
    log_matrix = np.log10(np.array(variance_matrix) + 1e-12)

    sns.heatmap(log_matrix, ax=ax, cmap='YlOrRd_r',
                xticklabels=layer_range, yticklabels=qubit_range,
                cbar_kws={'label': 'log₁₀(Gradient Variance)'})

    ax.set_xlabel('Number of Layers', fontsize=12)
    ax.set_ylabel('Number of Qubits', fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path / 'variance_heatmap.png', dpi=150)
    plt.close()


def plot_cost_landscape_slice(cost_values, param_range, labels=None,
                               save_dir='results'):
    """
    1D slices through the cost landscape for different circuit configurations.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6', '#F39C12']

    for i, costs in enumerate(cost_values):
        label = labels[i] if labels else f'Config {i}'
        ax.plot(param_range, costs, '-', color=colors[i % len(colors)],
                linewidth=2, label=label)

    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_ylabel('Cost Function Value', fontsize=12)
    ax.set_title('Cost Landscape Slices')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'cost_landscape_slices.png', dpi=150)
    plt.close()


def plot_layerwise_convergence(results_dict, save_dir='results'):
    """
    Plot training convergence for layer-wise vs global training.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # cost history
    ax = axes[0]
    colors = {'Layer-wise': '#4ECDC4', 'Global': '#FF6B6B'}
    for name, data in results_dict.items():
        ax.plot(data['cost_history'], '-', color=colors.get(name, 'gray'),
                linewidth=1.5, alpha=0.8, label=name)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Cost Value')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # layer costs (for layerwise only)
    ax = axes[1]
    if 'Layer-wise' in results_dict and 'layer_costs' in results_dict['Layer-wise']:
        layer_costs = results_dict['Layer-wise']['layer_costs']
        ax.bar(range(1, len(layer_costs) + 1), layer_costs,
               color='#4ECDC4', alpha=0.8, edgecolor='#2C3E50')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cost After Training Layer')
        ax.set_title('Layer-wise Training Progress')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path / 'layerwise_convergence.png', dpi=150)
    plt.close()


def plot_global_vs_local_cost(global_results, local_results, save_dir='results'):
    """
    Compare gradient variance between global and local cost functions.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(global_results['n_qubits'], global_results['variances'],
                'o-', color='#FF6B6B', linewidth=2, markersize=8,
                label='Global Cost')
    ax.semilogy(local_results['n_qubits'], local_results['variances'],
                's-', color='#4ECDC4', linewidth=2, markersize=8,
                label='Local Cost')

    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Gradient Variance (log scale)', fontsize=12)
    ax.set_title('Global vs Local Cost Function: Gradient Scaling')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'global_vs_local_cost.png', dpi=150)
    plt.close()
